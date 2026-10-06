import json
import shutil
from pathlib import Path

import pytest

from servekit import cli
from servekit.decompose.replay import decompose, render
from servekit.decompose.schema import available_specs, load_schema, load_spec, select_spec

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = load_schema()
H100 = FIXTURES / "vllm-0.28.1rc1-glm-h100.log"
H100_LAUNCH = "2026-10-05T16:18:25.112827"
QWEN = FIXTURES / "vllm-0.18.0-qwen3.log"
QWEN_LAUNCH = "2026-10-01T14:25:49.333846"
APERTUS = FIXTURES / "vllm-0.14.1-apertus.log"
APERTUS_LAUNCH = "2026-10-01T12:24:00.495318"


def approx(value, expected, tol=0.06):
    assert value == pytest.approx(expected, abs=tol)


def write_spec(tmp_path, version="9.9.9", events=None, **extra):
    root = tmp_path / "vllm"
    root.mkdir(exist_ok=True)
    path = root / f"{version}.json"
    path.write_text(json.dumps({"engine": "vllm", "version": version, "events": events or [], **extra}))
    return path


GOOD_EVENTS = [{"line": "x", "phase": "post_load"}, {"line": "y", "phase": "final_warmup"}]


# --- spec files and validation ------------------------------------------------------------------

def test_every_shipped_spec_validates():
    paths = available_specs()
    assert len(paths) >= 3
    for path in paths:
        assert load_spec(path, SCHEMA).version == path.stem


@pytest.mark.parametrize("events, message", [
    ([{"line": "x", "phase": "no_such_phase"}], "not a schema phase"),
    ([{"line": "x", "phase": "post_load", "took": "([0-9.]+)", "before": "nope"}], "not a schema phase"),
    ([{"line": "(", "phase": "post_load"}], "bad regex"),
    ([{"line": "x", "phase": "post_load", "took": "([0-9.]+)"}], "go together"),
    ([{"line": "x", "phase": "post_load", "before": "pre_load"}], "go together"),
    ([{"line": "x", "phase": "post_load"}], "needs an event for phase"),  # no final_warmup for the engine check
])
def test_validator_rejects_bad_specs(tmp_path, events, message):
    with pytest.raises(ValueError, match=message):
        load_spec(write_spec(tmp_path, events=events), SCHEMA)


def test_validator_rejects_file_name_that_disagrees_with_version(tmp_path):
    path = write_spec(tmp_path, events=GOOD_EVENTS)
    renamed = path.with_name("1.2.3.json")
    path.rename(renamed)
    with pytest.raises(ValueError, match="does not match version"):
        load_spec(renamed, SCHEMA)


# --- choosing a spec ----------------------------------------------------------------------------

@pytest.mark.parametrize("version, expected, approximate", [
    ("0.14.1", "0.14.1", False),
    ("0.18.0", "0.18.0", False),
    ("0.28.1rc1.dev580+g385dce36b", "0.28.1rc1.dev580+g385dce36b", False),
    ("0.18.1", "0.18.0", True),
    ("0.28.1rc1.dev580+gdifferent", "0.28.1rc1.dev580+g385dce36b", True),
    ("0.30.0", "0.28.1rc1.dev580+g385dce36b", True),
    ("0.24.0+092c4842.dev", "0.18.0", True),
])
def test_select_spec(version, expected, approximate):
    spec, flagged = select_spec(version, SCHEMA)
    assert (spec.version, flagged) == (expected, approximate)


def test_version_older_than_every_spec_is_refused():
    with pytest.raises(ValueError, match="no vllm spec at or below 0.10.0"):
        select_spec("0.10.0", SCHEMA)


# --- real logs ----------------------------------------------------------------------------------

def check_closes(report):
    assert sum(v for v in report["phases"].values() if v is not None) == pytest.approx(report["total_s"], abs=0.1)
    assert [c for c in report["checks"] if c["ok"] is not True] == []


def test_h100_glm_log_attributes_every_second():
    report = decompose(H100, launch_time=H100_LAUNCH)
    assert report["spec"] == "0.28.1rc1.dev580+g385dce36b" and not report["spec_approximate"]
    check_closes(report)
    p = report["phases"]
    approx(report["total_s"], 650.63)
    approx(p["container_start"], 41.55)
    approx(p["weight_loading"], 84.02)
    approx(p["deepgemm_warmup"], 191.84)
    approx(p["graph_capture"], 146.0)
    assert [x["duration_s"] for x in report["timeline"] if x["detail"]] == [142.0, 4.0]
    assert p["dynamo_transform"] is None and p["compile_backend"] is None and p["initial_profile_run"] is None
    approx(report["checks"][0]["reported_s"], 385.0)
    approx(report["checks"][0]["diff_s"], 0.0, 0.1)


def test_vllm_018_log_reads_took_compile_line():
    report = decompose(QWEN, launch_time=QWEN_LAUNCH)
    assert report["spec"] == "0.18.0"
    check_closes(report)
    p = report["phases"]
    approx(p["dynamo_transform"], 13.11)
    approx(p["compile_backend"], 20.16)
    approx(p["initial_profile_run"], 1.09)
    assert p["deepgemm_warmup"] is None
    # CUDA graph memory profiling is its own phase; the KV cache itself is sized in well under a second
    approx(p["graph_memory_profiling"], 1.66)
    assert p["kv_sizing"] < 1.0


def test_vllm_014_log_reads_takes_compile_line_and_colour_codes():
    report = decompose(APERTUS, launch_time=APERTUS_LAUNCH)
    assert report["spec"] == "0.14.1"
    check_closes(report)
    approx(report["phases"]["compile_backend"], 21.45)
    assert report["phases"]["initial_profile_run"] is None


def test_without_launch_time_container_start_is_not_measured():
    report = decompose(H100)
    assert report["phases"]["container_start"] is None
    assert report["interval_start"] == "first log line"
    assert any("--launch-time" in n for n in report["notes"])
    check_closes(report)
    approx(report["total_s"], 609.08)


def test_launch_time_after_first_log_line_is_rejected():
    with pytest.raises(ValueError, match="after the first log line"):
        decompose(H100, launch_time="2026-10-05T16:30:00")


def test_vllm_own_timestamps_and_unknown_version():
    """The existing fixture has no Docker prefix and is vLLM 0.24.0, which has no spec file."""
    report = decompose(FIXTURES / "llama70b-vllm-ngc.log", year=2026)
    assert report["version"].startswith("0.24.0") and report["spec_approximate"]
    assert report["spec"] == "0.18.0"
    check_closes(report)
    assert "approximate" in render(report)


def test_a_missing_line_is_reported_not_swallowed(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("".join(l for l in H100.read_text().splitlines(keepends=True) if "DeepGEMM warmup" not in l))
    report = decompose(log, launch_time=H100_LAUNCH)
    assert any("DeepGEMM" in e for e in report["events_not_seen"])
    check_closes(report)  # the time is still accounted for, just under the next phase


def test_log_without_timestamps_is_refused(tmp_path):
    log = tmp_path / "plain.log"
    log.write_text("hello\nworld\n")
    with pytest.raises(ValueError, match="no timestamps"):
        decompose(log)


def test_log_that_never_became_ready_is_refused(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("".join(l for l in H100.read_text().splitlines(keepends=True) if "Application startup complete" not in l))
    with pytest.raises(ValueError, match="never reached ready"):
        decompose(log)


# --- command line -------------------------------------------------------------------------------

def test_cli_log_mode_prints_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli.main(["profile", "--log", str(QWEN), "--launch-time", QWEN_LAUNCH, "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "weight_loading" in text and "check engine_init" in text
    assert json.loads(out.read_text())["spec"] == "0.18.0"


def test_cli_log_mode_reports_unreadable_log(tmp_path, capsys):
    assert cli.main(["profile", "--log", str(tmp_path / "missing.log")]) == 2


# --- rendering ----------------------------------------------------------------------------------

def test_table_omits_phases_the_version_lacks():
    text = render(decompose(H100, launch_time=H100_LAUNCH))
    rows = {line.split()[0] for line in text.splitlines() if line and line.split()[0] in SCHEMA.leaves}
    assert "dynamo_transform" not in rows and "compile_backend" not in rows and "deepgemm_warmup" in rows
    assert "not in this vLLM version: compile_setup, dynamo_transform, compile_backend, initial_profile_run" in text
    assert "n/a" not in text


def test_show_na_lists_the_missing_phases_as_rows():
    text = render(decompose(H100, launch_time=H100_LAUNCH), show_na=True)
    assert "dynamo_transform" in text and "n/a" in text and "not in this vLLM version" not in text


def test_missing_launch_time_is_a_note_not_a_missing_phase():
    text = render(decompose(H100))
    assert "container_start not measured" in text and "container_start" not in text.split("not in this vLLM version:")[-1].splitlines()[0]


def test_cli_json_prints_only_the_report(capsys):
    code = cli.main(["profile", "--log", str(QWEN), "--launch-time", QWEN_LAUNCH, "--json"])
    report = json.loads(capsys.readouterr().out)  # the whole of stdout is one JSON document
    assert code == 0 and report["spec"] == "0.18.0" and report["phases"]["weight_loading"] > 0
