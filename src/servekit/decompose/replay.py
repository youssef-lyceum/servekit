"""Decompose a saved engine log into schema phases, using the spec for the log's own version."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import Schema, Spec, load_schema, select_spec

DOCKER_STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z ")
# vLLM stamps mid-line without a year: "INFO 07-28 17:04:47 [monitor.py:53]".
VLLM_STAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
BANNER = re.compile(r"version (\d+\.\d+\.\d+[0-9A-Za-z.+_-]*)")
WORLD_SIZE = re.compile(r"\bworld_size=(\d+)")
TENSOR_PARALLEL = re.compile(r"'tensor_parallel_size': (\d+)")  # vLLM prints only non-default launch arguments
CLUSTER_GAP_S = 5.0
READY_PHASE = "server_startup"


def _epoch(iso: str) -> float:
    head, _, frac = iso.partition(".")
    base = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return base + (float("0." + frac) if frac else 0.0)


def parse_launch_time(value: str) -> float:
    return _epoch(value.rstrip("Z").replace(" ", "T"))


def read_lines(path: Path, year: Optional[int] = None) -> List[Tuple[float, str]]:
    """(time, text) per line. Unstamped continuation lines (progress bars) inherit the previous time."""
    year = year or datetime.now().year
    out: List[Tuple[float, str]] = []
    last: Optional[float] = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        m = DOCKER_STAMP.match(raw)
        if m:
            last, raw = _epoch(m.group(1)), raw[m.end():]
        else:
            m = VLLM_STAMP.search(raw)
            if m:
                month, day, hh, mm, ss = (int(g) for g in m.groups())
                last = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc).timestamp()
        if last is not None:
            out.append((last, ANSI.sub("", raw)))
    if not out:
        raise ValueError(f"{path}: no timestamps found (expected Docker `-t` prefixes or vLLM 'MM-DD HH:MM:SS' stamps)")
    return out


def find_version(lines: List[Tuple[float, str]]) -> str:
    for _, text in lines[:400]:
        m = BANNER.search(text)
        if m:
            return m.group(1)
    raise ValueError("no vLLM version banner in the first 400 lines; pass the version explicitly")


def find_parallelism(lines: List[Tuple[float, str]]) -> Dict[str, Optional[int]]:
    """World size from the distributed-init lines; tensor parallel size from the non-default arguments line.
    vLLM omits arguments left at their default, so with a world size of 1 an absent value means 1. With more GPUs and
    no value it could be pipeline or data parallelism, so it stays unknown."""
    sizes = [int(m.group(1)) for _, text in lines if (m := WORLD_SIZE.search(text))]
    world = max(sizes) if sizes else None
    tp = next((int(m.group(1)) for _, text in lines if (m := TENSOR_PARALLEL.search(text))), None)
    if tp is None and world == 1:
        tp = 1
    return {"world_size": world, "tensor_parallel_size": tp}


def _clusters(hits: List[Tuple[float, str]]) -> List[List[Tuple[float, str]]]:
    groups: List[List[Tuple[float, str]]] = []
    for hit in hits:
        if groups and hit[0] - groups[-1][-1][0] <= CLUSTER_GAP_S:
            groups[-1].append(hit)
        else:
            groups.append([hit])
    return groups


def decompose(log: Path, launch_time: Optional[str] = None, year: Optional[int] = None,
              version: Optional[str] = None, schema: Optional[Schema] = None,
              specs_root: Optional[Path] = None) -> Dict:
    schema = schema or load_schema()
    lines = read_lines(Path(log), year)
    version = version or find_version(lines)
    kw = {"root": specs_root} if specs_root else {}
    spec, approximate = select_spec(version, schema, **kw)

    # One event per cluster of matching lines, taken at its last line. Same-phase events add up.
    observed: List[Tuple[float, int, str, Optional[float], Optional[str], Optional[str]]] = []
    not_seen: List[str] = []
    for order, ev in enumerate(spec.events):
        hits = [(t, text) for t, text in lines if ev.line.search(text)]
        if not hits:
            not_seen.append(ev.line.pattern)
            continue
        groups = _clusters(hits)
        for n, group in enumerate(groups, 1):
            took = None
            if ev.took:
                vals = [float(m.group(1)) for _, text in group if (m := ev.took.search(text))]
                took = max(vals) if vals else None
            detail = f"pass {n}" if len(groups) > 1 and ev.took else None
            observed.append((group[-1][0], order, ev.phase, took, ev.before, detail))

    ready = [o for o in observed if o[2] == READY_PHASE]
    if not ready:
        raise ValueError(f"{log}: never reached ready (no '{READY_PHASE}' event)")
    end = min(o[0] for o in ready)
    observed = sorted((o for o in observed if o[0] <= end), key=lambda o: (o[0], o[1]))

    notes: List[str] = []
    first_line = lines[0][0]
    if launch_time:
        start = parse_launch_time(launch_time)
        if start > first_line:
            raise ValueError(f"--launch-time {launch_time} is after the first log line")
        observed.insert(0, (first_line, -1, "container_start", None, None, None))
        interval_start = "launch time"
    else:
        start = first_line
        interval_start = "first log line"
        notes.append("container_start not measured: no --launch-time, so the interval starts at the first log line")

    timeline: List[Dict] = []
    prev = start
    for t, _, phase, took, before, detail in observed:
        gap = max(t - prev, 0.0)
        if took is not None and 0 < took < gap:  # the engine says the phase took `took`; the rest came before it
            timeline.append(dict(start_s=round(prev - start, 2), duration_s=round(gap - took, 2), phase=before, detail=None))
            prev += gap - took
            gap = took
        timeline.append(dict(start_s=round(prev - start, 2), duration_s=round(gap, 2), phase=phase, detail=detail))
        prev = max(prev, t)

    used = {e.phase for e in spec.events} | {e.before for e in spec.events if e.before}
    if launch_time:
        used.add("container_start")
    phases: Dict[str, Optional[float]] = {}
    for leaf in schema.leaves:
        phases[leaf] = round(sum(p["duration_s"] for p in timeline if p["phase"] == leaf), 2) if leaf in used else None

    checks = [_check(c, observed, lines, start) for c in schema.checks]
    return dict(
        log=str(log), engine=spec.engine, version=version, **find_parallelism(lines),
        spec=spec.version, spec_approximate=approximate,
        interval_start=interval_start, total_s=round(end - start, 2), phases=phases, timeline=timeline,
        events_not_seen=not_seen, checks=checks, notes=notes)


def _check(check: dict, observed, lines, start: float) -> Dict:
    pattern = re.compile(check["reported_match"])
    reported = [float(m.group(1)) for _, text in lines if (m := pattern.search(text))]
    ends = {phase: max((o[0] for o in observed if o[2] == phase), default=None)
            for phase in (check["from_phase"], check["to_phase"])}
    result = dict(id=check["id"], ok=None, reported_s=max(reported) if reported else None, from_phases_s=None, diff_s=None)
    if not reported or None in ends.values():
        result["skipped"] = "engine did not report the figure" if not reported else "span events not seen"
        return result
    ours = ends[check["to_phase"]] - ends[check["from_phase"]]
    result.update(from_phases_s=round(ours, 2), diff_s=round(ours - max(reported), 2))
    result["ok"] = abs(ours - max(reported)) <= check["tolerance_s"]
    return result


def render(report: Dict, schema: Optional[Schema] = None, show_na: bool = False) -> str:
    """Table of the phases this version has. Phases it does not print are named in one line unless show_na."""
    schema = schema or load_schema()
    width = max(len(k) for k in schema.leaves)
    approx = "  (approximate: no spec for this exact version)" if report["spec_approximate"] else ""
    phases = report["phases"]
    out = [f"{report['engine']} {report['version']}  spec={report['spec']}{approx}  interval starts at: {report['interval_start']}",
           f"{'phase':<{width}}  {'seconds':>9}", "-" * (width + 11)]
    detail: Dict[str, List[str]] = {}
    for p in report["timeline"]:
        if p["detail"]:
            detail.setdefault(p["phase"], []).append(f"{p['detail']}: {p['duration_s']:.2f}")
    for leaf, secs in phases.items():
        if secs is None:
            if show_na:
                out.append(f"{leaf:<{width}}  {'n/a':>9}")
            continue
        out.append(f"{leaf:<{width}}  {secs:9.2f}")
        for d in detail.get(leaf, []):
            out.append(f"{'':<{width}}    {d}")
    out += ["-" * (width + 11), f"{'total':<{width}}  {report['total_s']:>9.2f}"]
    absent = [leaf for leaf, secs in phases.items() if secs is None and leaf != "container_start"]
    if absent and not show_na:
        out.append("not in this vLLM version: " + ", ".join(absent))
    out += [f"note: {n}" for n in report["notes"]]
    if report["events_not_seen"]:
        out.append("events not seen: " + "; ".join(report["events_not_seen"]))
    for c in report["checks"]:
        if c.get("skipped"):
            out.append(f"check {c['id']}: skipped ({c['skipped']})")
        else:
            out.append(f"check {c['id']}: engine reports {c['reported_s']} s, phases give {c['from_phases_s']} s "
                       f"(diff {c['diff_s']:+.2f}) {'ok' if c['ok'] else 'MISMATCH'}")
    return "\n".join(out)
