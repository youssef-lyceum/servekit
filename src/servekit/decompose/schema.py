"""Phase schema and per-release specs for decomposing an engine's startup log."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).parent
VERSION_PREFIX = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?(?:\.dev(\d+))?")


@dataclass(frozen=True)
class Event:
    line: "re.Pattern"
    phase: str
    took: Optional["re.Pattern"] = None
    before: Optional[str] = None


@dataclass(frozen=True)
class Spec:
    engine: str
    version: str
    path: Path
    events: Tuple[Event, ...]


@dataclass(frozen=True)
class Schema:
    leaves: Dict[str, str]
    checks: Tuple[dict, ...]


def load_schema(path: Path = HERE / "schema.json") -> Schema:
    data = json.loads(path.read_text())
    return Schema(leaves=data["leaves"], checks=tuple(data["checks"]))


def _compile(pattern: str, where: str) -> "re.Pattern":
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"{where}: bad regex {pattern!r}: {e}") from None


def load_spec(path: Path, schema: Schema) -> Spec:
    """Read one release file and refuse it if it disagrees with the schema."""
    data = json.loads(path.read_text())
    version = data["version"]
    if path.stem != version:
        raise ValueError(f"{path.name}: file name does not match version {version!r}")
    events: List[Event] = []
    for i, raw in enumerate(data["events"]):
        where = f"{path.name} event {i} ({raw.get('line')!r})"
        for leaf in (raw["phase"], raw.get("before")):
            if leaf is not None and leaf not in schema.leaves:
                raise ValueError(f"{where}: {leaf!r} is not a schema phase")
        if bool(raw.get("took")) != bool(raw.get("before")):
            raise ValueError(f"{where}: took and before go together")
        events.append(Event(
            line=_compile(raw["line"], where),
            phase=raw["phase"],
            took=_compile(raw["took"], where) if raw.get("took") else None,
            before=raw.get("before"),
        ))
    phases = {e.phase for e in events}
    for check in schema.checks:
        for key in ("from_phase", "to_phase"):
            if check[key] not in phases:
                raise ValueError(f"{path.name}: check {check['id']!r} needs an event for phase {check[key]!r}")
    return Spec(engine=data["engine"], version=version, path=path, events=tuple(events))


def version_key(version: str) -> Tuple[int, ...]:
    """Orderable key for major.minor.patch[rcN][.devN]. A dev build sorts before its rc, an rc before the release.
    The +g<commit> suffix is ignored, so two files can share a key."""
    m = VERSION_PREFIX.match(version)
    if not m:
        raise ValueError(f"cannot parse vLLM version {version!r}")
    major, minor, patch, rc, dev = m.groups()
    return (int(major), int(minor), int(patch),
            1 if rc is None else 0, int(rc or 0),
            1 if dev is None else 0, int(dev or 0))


def available_specs(engine: str = "vllm", root: Path = HERE / "specs") -> List[Path]:
    return sorted((root / engine).glob("*.json"))


def select_spec(version: str, schema: Schema, engine: str = "vllm", root: Path = HERE / "specs") -> Tuple[Spec, bool]:
    """Exact file for `version`, else the nearest older one. Second value is True when it is not exact."""
    wanted = version_key(version)
    best: Optional[Path] = None
    for path in available_specs(engine, root):
        if path.stem == version:
            return load_spec(path, schema), False
        if version_key(path.stem) <= wanted and (best is None or version_key(path.stem) > version_key(best.stem)):
            best = path
    if best is None:
        known = ", ".join(p.stem for p in available_specs(engine, root)) or "none"
        raise ValueError(f"no {engine} spec at or below {version} (have: {known})")
    return load_spec(best, schema), True
