#yfrom __future__ import annotations

import re, json
import subprocess
from typing import Dict, Any, List, Iterable
import functools
import contextlib
import io
import itertools
import pathlib
from pathlib import Path
import sys
import os


try:
    from adk import tool  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tool(**_kw):  # type: ignore
        def deco(fn):
            return fn
        return deco

try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    raise ImportError("`get_swe_lite_instance` requires `pip install datasets`.")

def _run(cmd: list[str], cwd: pathlib.Path) -> None:
    """Quiet Git wrapper, raises on non-zero exit."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(res.stderr or res.stdout or "git failed")

def prepare_repo(repo_slug: str, commit: str, workspace_root: str = "/tmp") -> str:
    src = pathlib.Path(workspace_root) / repo_slug.replace("/", "__") / "src"
    src.mkdir(parents=True, exist_ok=True)

    # Init a brand-new repo
    subprocess.run(["git", "init"], cwd=src, check=True, capture_output=True)
    subprocess.run([
        "git", "remote", "add", "origin", f"https://github.com/{repo_slug}.git"
    ], cwd=src, check=True, capture_output=True)

    # Shallow-fetch just this one commit
    subprocess.run([
        "git", "fetch", "--depth", "1", "origin", commit
    ], cwd=src, check=True, capture_output=True)

    # And check it out
    subprocess.run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=src, check=True)

    return str(src.resolve())

def _count_changed_lines(diff: str) -> tuple[int, int]:
    """Return (#added, #deleted) excluding hunk headers."""
    added = len(
        [ln for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    )
    deleted = len(
        [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    )
    return added, deleted


@tool(
    name="ApplyPatch",
    description="Apply a unified diff to the repo and return whether it applied "
                "cleanly. Rejects multi-line edits by default (set allow_multi_line=True "
                "to override).",
)
def apply_patch(diff: str) -> Dict[str, str | int | bool]:
    """Apply *diff* with `git apply`.

    Parameters
    ----------
    diff : str
        Unified diff produced by `git diff -U<n>` or similar.
    """
    print("apply_patch called")
    added, deleted = _count_changed_lines(diff)
    total_changes = added + deleted
    if total_changes > 1:
        return {
            "ok": False,
            "msg": f"patch touches {total_changes} lines (limit is 1)",
            "lines_added": added,
            "lines_deleted": deleted,
        }

    # Run git apply with whitespace warnings suppressed (common in
    # third‑party codebases)
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=diff.encode(),
        capture_output=True,
    )

    if proc.returncode == 0:
        return {
            "ok": True,
            "msg": "applied cleanly",
            "lines_added": added,
            "lines_deleted": deleted,
        }

    # Failure – attempt to revert (in case of partial apply)
    subprocess.run(["git", "apply", "-R", "-"], input=diff.encode())
    err = proc.stderr.decode(errors="ignore").strip() or "git apply failed"
    return {"ok": False, "msg": err, "lines_added": added, "lines_deleted": deleted}


@functools.lru_cache(maxsize=3)
def _load_split(split: str):
    return load_dataset("princeton-nlp/SWE-bench_Lite", split=split)


@tool(
    name="GetSweBenchLiteInstance",
    description="Return the dictionary row from the SWE-bench Lite dataset that"
                " matches the given instance_id ",
)
def get_swe_lite_instance(instance_id: str) -> Dict[str, Any]:
    """Strictly fetch one row from *SWE-bench_Lite* by instance_id."""

    ds = _load_split("test")

    # The arrow format keeps a column `instance_id`; use dataset filtering
    matches = ds.filter(lambda x: x["instance_id"] == instance_id)
    if len(matches) == 0:
        raise ValueError(
            f"instance_id '{instance_id}' not found in SWE-bench_Lite"
        )

    # There should be only one match
    row = matches[0]
    return dict(row)  # convert to raw Python dict for ADK serialisation


@tool(
    name="GitReset",
    description="Rollback all workspace changes. Equivalent to 'git reset --hard'"
                " followed by optional 'git clean -fd'. Pass a specific commit"
                " hash to reset to that point.",
)
def git_reset(commit: str = "--hard", clean_untracked: bool = True) -> Dict[str, str | bool]:
    """Reset the git working tree.

    Parameters
    ----------
    commit : str, default "--hard"
        The target for `git reset --hard`.  Common options:
        * ``"--hard"`` – shorthand for current *HEAD*, keep branch ref intact.
        * commit hash – checkout that revision (used when you want to go back
          to *base_commit* after installing deps at *environment_setup_commit*).
    clean_untracked : bool, default True
        Whether to also run ``git clean -fd`` to delete untracked files.
    """

    # Step 1 – reset --hard …
    proc = subprocess.run(
        ["git", "reset", "--hard", commit], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {"ok": False, "msg": proc.stderr.strip() or "git reset failed"}

    msg = proc.stdout.strip() or proc.stderr.strip()

    # Step 2 – clean untracked files/dirs
    if clean_untracked:
        clean_proc = subprocess.run(
            ["git", "clean", "-fd"], capture_output=True, text=True
        )
        if clean_proc.returncode != 0:
            return {
                "ok": False,
                "msg": clean_proc.stderr.strip() or "git clean failed",
            }
        if clean_proc.stdout:
            msg += "\n" + clean_proc.stdout.strip()

    # Retrieve new HEAD hash for convenience
    head_proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    head_hash = head_proc.stdout.strip() if head_proc.returncode == 0 else "unknown"

    return {"ok": True, "msg": msg, "head": head_hash}


@tool(
    name="search_code",
    description="Search for the pattern through the repository and return matching "
                "file, line number, and snippet (up to `max_results`)."
)
def search_code(
    pattern: str,
    path: List[str] ,
) -> List[Dict]:
    """
    Find single-line matches.

    Returns list of:
        {path: str, line: int, snippet: str}
    """
    roots = list(path)
    return (_with_rg if _has_ripgrep() else _with_py)(pattern, roots, 1)


def _has_ripgrep() -> bool:
    return subprocess.call(
        ["which", "rg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) == 0


def _with_rg(pattern: str, roots: List[str], limit: int) -> List[Dict]:
    cmd = [
        "rg",
        "--json",
        "--line-number",
        "--max-count",
        str(limit) if limit else "0",
        pattern,
        *roots,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode not in (0, 1):
        raise RuntimeError(out.stderr.strip())

    hits: List[Dict] = []
    for raw in out.stdout.splitlines():
        ev = json.loads(raw)
        if ev.get("type") != "match":
            continue
        d = ev["data"]
        hits.append(
            {
                "path": d["path"]["text"],
                "line": d["line_number"],
                "snippet": d["lines"]["text"].rstrip("\n"),
            }
        )
        if limit and len(hits) >= limit:
            break
    return hits


def _with_py(pattern: str, roots: List[str], limit: int) -> List[Dict]:
    hits: List[Dict] = []
    for root in roots:
        for file in Path(root).rglob("*"):
            if not file.is_file():
                continue
            try:
                for i, line in enumerate(
                    file.open(encoding="utf-8", errors="ignore"), 1
                ):
                    if pattern in line:
                        hits.append(
                            {"path": str(file), "line": i, "snippet": line.rstrip("\n")}
                        )
                        if limit and len(hits) >= limit:
                            return hits
            except Exception:
                pass
    return hits
@tool(
    name="RunTests",
    description="Execute pytest and return a summary dict with exit_code, passed,"
                " failed counts, and a truncated log (max_output chars).",
)

def run_tests(paths: str = "", max_output: int = 20000) -> Dict[str, str | int]:
    """Run pytest programmatically.

    Parameters
    ----------
    paths : str
        Space‑separated list of test paths. Empty string runs the full suite.
    max_output : int
        Cap the size of the returned log to avoid blowing the token budget.
    """
    print("run_tests called")
    import pytest  # local import keeps module import cheap if pytest heavy

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exit_code = pytest.main(paths.split())
    log = buf.getvalue()

    # Parse summary line(s)
    fail = pass_ = 0
    m = re.search(r"= *(\d+) failed", log)
    if m:
        fail = int(m.group(1))
    m = re.search(r"= *(\d+) passed", log)
    if m:
        pass_ = int(m.group(1))

    # Truncate long logs (keep head + tail)
    if len(log) > max_output:
        half = max_output // 2
        log = log[:half] + "\n[…truncated…]\n" + log[-half:]

    return {
        "exit_code": exit_code,
        "passed": pass_,
        "failed": fail,
        "log": log,
    }

__all__ = [
    "search_code",
    "run_tests",
    "apply_patch",
    "git_reset",
    "get_swe_lite_instance",
    "prepare_repo",
]