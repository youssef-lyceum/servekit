# Servekit 🧊 → 🔥

Servekit is a kit of optimizations for serving large language models (LLMs) with minimal cold-start latency 🥶. It wraps sglang calls and implements some optimizations.

Currently, it supports fast weight loading for **lustre** storages like CSCS's `capstor` and `iopsstor` ⚡.


## Performance ⚡

Measured with `servekit profile`/`servekit bench` on CSCS Clariden (GH200), SGLang
v0.5.10.

| Model | Setup | Loader | `weight_loading` | Total cold start |
| --- | --- | --- | --- | --- |
| Llama-3.1-70B | TP4, 1 node | default | 466.81s | 586.33s |
| | | **servekit** | **6.19s** | **127.06s** |
| Llama-3.1-70B | TP8, 2 nodes | default | 553.02s | 667.44s |
| | | **servekit** | **3.25s** | **123.40s** |
| Apertus-8B | TP4, 1 node | default | 81.41s | 172.68s |
| | | **servekit** | **0.90s** | **95.53s** |

Servekit achieves faster weight loading by **75x-170x**, and an overall faster cold start by **1.8x-5.4x**.

## Setup

Install directly from GitHub:

```bash
pip install git+https://github.com/eth-easl/servekit.git
```

Or, for local development (editable install, from this directory):

```bash
pip install -e .
```

## Usage

### `servekit launch`

```bash
servekit launch --servekit-artifact-path <dir> \
  -- python -m sglang.launch_server --model-path <model> --tensor-parallel-size 4 ...
```

Prepend `servekit launch --servekit-artifact-path <dir> --` to an engine command
to enable servekit's optimizations. Currently, we offer:
* Fast weight loading: loads the weights in a multiprocessing fashion adapted to network storage like Lustre drives (e.g. CSCS `capstor` and `iopsstor`) to RAM (`/dev/shm`) and then to GPU memory.

`<dir>` is where servekit keeps its presharded copy of the checkpoint. It writes
one if the directory has none, so `--model-path` always names the real model. If
the artifact doesn't fit the command, servekit says which setting disagrees and
falls back to the engine's own loader rather than failing the job.

`--only-prepare` writes the artifact and stops. Pipeline parallelism needs
sglang >= 0.5.14 (`sglang.srt.runtime_context`, for pp_rank/pp_size).

`--overlap` starts the engine while the stage is still running, hiding it behind
engine startup; sglang only.


### `servekit profile`

```bash
servekit profile -- python -m sglang.launch_server --model-path <model> ...
servekit profile -- vllm serve <model> --tensor-parallel-size 4 ...
```

Parses the engine's own log output for phase timings (no engine changes
needed) and prints a per-phase duration table once the server is ready. 


### `servekit profile --log`

```bash
servekit profile --log server.log [--launch-time 2026-10-05T16:18:25] [--out report.json]
```

Decomposes a *saved* engine log (vLLM) into a fixed list of phases, so runs from different vLLM
versions line up. Phases come from `src/servekit/decompose/schema.json`; the log lines that close each
phase come from one self-contained file per vLLM release in `src/servekit/decompose/specs/vllm/`.

* **Version.** Read from the log's own banner. An exact file is used if there is one; otherwise the
  nearest older file, and the output says "approximate". A version older than every file is refused.
* **Timestamps.** Docker `-t` prefixes, or vLLM's own `MM-DD HH:MM:SS` stamps (`--year`, default the
  current year; those have 1 s resolution).
* **`--launch-time`** is optional and is not in the log. It is the UTC time the container started, for
  example `docker inspect <container> --format '{{.State.StartedAt}}'` recorded at run time. With it,
  `container_start` (Docker start plus imports, 41 s on an 8xH100 GLM run) is counted; without it the
  interval starts at the first log line and `container_start` shows `n/a`.
* **Checks.** Time between the end of `post_load` and the end of `final_warmup` must equal the engine's
  own `init engine ... took` figure (within 0.25 s), and the exit code is 1 if it does not.
  Spec lines that never appear in the log are listed as "events not seen", which is how a renamed line shows up.
* **Output.** One row per phase the version has, per-pass rows under `graph_capture`, and the total.
  Phases the version does not print (for example `torch.compile` on 0.28) are named in one
  "not in this vLLM version" line instead of empty rows; `--show-na` lists them as rows.
  `--out FILE` writes the report as JSON, where absent phases are `null`; `--json` prints that JSON on stdout
  instead of the table, so it can be piped (`servekit profile --log x.log --json | jq .phases`).

To support a new release, copy the closest file in `specs/vllm/`, rename it to the exact version,
adjust the `line` regexes, and run `pytest tests/test_decompose.py`. Each event is
`{"line": regex, "phase": name}`; an engine that prints a phase's own duration adds
`"took": regex-with-one-number` and `"before": phase` (the name for the time before that phase started).
Matching lines within 5 s of each other count as one event, and events with the same phase add up.

### `servekit bench`

```bash
servekit bench --url http://127.0.0.1:8080 --out bench.json --wait-ready 300
```

Runs a correctness check (greedy completions on fixed prompts) and a fixed
concurrent throughput workload against `POST /v1/completions`. 

### `servekit verify`

Checks that a served model produces the same numbers as a trusted reference —
per-token logprobs, mean NLL, and greedy continuations over a fixed prompt set.

```bash
# record a reference from a server you trust (e.g. plain sglang, no servekit)
servekit verify --url http://127.0.0.1:8080 --record gold.json --wait-ready 300

# check a later server (e.g. one started with `servekit launch`) against it
servekit verify --url http://127.0.0.1:8080 --reference gold.json --wait-ready 300
```

Exits 0 if every prompt is within tolerance (`--token-tol`, `--nll-tol`), 1
otherwise. `--out` writes the per-prompt result as JSON.

### Roadmap 👷‍♂️🚧

- [x] Support Multi-Node fast weight loading
- [x] Support Pipeline parallelism (needs sglang >= 0.5.14)
- [ ] Support vllm fast weight loading

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```

The e2e suite needs a live SLURM cluster and runs only when asked for:

```bash
PYTHONPATH=src python -m pytest tests/e2e -m e2e -q
```
