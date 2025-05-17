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
import difflib

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

def prepare_repo(
    repo_slug: str,
    commit: str,
    workspace_root: str = "/tmp",
    mirror_root: str = "/mirror",
) -> Dict[str, str]:
    """
    Returns {
      'repo_path':   '/tmp/<repo_slug>/src',
      'python_exe':  '/mirror/venvs/<repo_slug>/bin/python'
    }
    """
    # 1) Shallow‐clone into workspace_root/<slug>/src
    slug = repo_slug.replace("/", "__")
    work_src = pathlib.Path(workspace_root) / slug / "src"
    work_src.mkdir(parents=True, exist_ok=True)

    for cmd in (
        ["git", "init"],
        ["git", "remote", "add", "origin", f"https://github.com/{repo_slug}.git"],
        ["git", "fetch", "--depth", "1", "origin", commit],
        ["git", "checkout", "--quiet", "FETCH_HEAD"],
    ):
        subprocess.run(cmd, cwd=work_src, check=True, capture_output=True)

    # 2) Set up or reuse the per-repo venv under mirror_root/venvs/<slug>
    python_exe = setup_env(slug, mirror_root)

    return {
        "repo_path":  str(work_src.resolve()),
        "python_exe": python_exe,
    }

def setup_env(slug: str, mirror_root: str) -> str:
    """
    mirror_root is the absolute path inside the container (e.g. '/mirror').
    slug is the repo_slug.replace('/', '__') used above.
    Returns path to the venv's python binary.
    """
    mirror = pathlib.Path(mirror_root)
    venv_root = mirror / "venvs" / slug
    python_bin = venv_root / "bin" / "python"
    pip_bin    = venv_root / "bin" / "pip"

    if not venv_root.exists():
        # 1) Create the venv
        subprocess.run(["python3", "-m", "venv", str(venv_root)], check=True)
        # 2) Install the repo's requirements.txt if present
        req = pathlib.Path("/tmp") / slug / "src" / "requirements.txt"
        if req.is_file():
            subprocess.run([str(pip_bin), "install", "-r", str(req)], check=True)

    return str(python_bin)

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
    name="apply_patch",
    description=(
        "Replace the line at `line_number` in `file_path` with `patch_text`, "
        "returning the unified diff."
    )
)
def apply_patch(
    patch_text: str,
    file_path: str,
    line_number: int
) -> Dict[str, Any]:
    """
    patch_text: the new code to insert (can be multiple lines, without leading/trailing newlines)
    file_path:  path to the file to modify (relative or absolute)
    line_number: 1-based line index to replace
    """
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    # Read original
    original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    idx = line_number - 1
    if not (0 <= idx < len(original_lines)):
        raise IndexError(f"line_number {line_number} out of range (1–{len(original_lines)})")

    # Prepare patched lines
    # Ensure final newline
    txt = patch_text
    if not txt.endswith("\n"):
        txt += "\n"
    patch_lines = txt.splitlines(keepends=True)

    # Build new file contents
    new_lines = original_lines[:idx] + patch_lines + original_lines[idx+1:]

    # Compute unified diff
    diff = "".join(difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=str(path),
        tofile=str(path),
        lineterm=""
    ))

    # Apply the change
    path.write_text("".join(new_lines), encoding="utf-8")

    return {
        "file": str(path),
        "line_number": line_number,
        "diff": diff,
    }
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
    name="git_reset",
    description="Rollback all workspace changes. Equivalent to 'git reset --hard'"
                " followed by optional 'git clean -fd'. Pass a specific commit"
                " hash to reset to that point.",
)
def git_reset(commit: str , clean_untracked: bool = True) -> Dict[str, str | bool]:
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
    name="run_tests",
    description="Execute pytest and return a summary dict with exit_code, passed,"
                " failed counts, and a truncated log (max_output chars).",
)

def run_tests(repo_path: str, paths: str, python_exe: str) -> Dict[str, Any]:
    """
    Parameters
    ----------
    repo_path : str
        Filesystem path to the checked-out repository.
    paths : str
        Space-separated list of test files or directories to run;
        if empty, runs the full suite.
    python_exe : str
    """
    cwd = pathlib.Path(repo_path)
    args = ["pytest", "-q", "--maxfail=1"]
    if paths:
        args += paths.split()

    # Launch pytest as a subprocess so plugin errors become exit codes
    proc = subprocess.run(
    [python_exe, "-m", "pytest", "-q", "--maxfail=1"] + paths.split(),
    cwd=cwd,
    capture_output=True,
    text=True,
)   

    out = proc.stdout + proc.stderr
    exit_code = proc.returncode

    # Extract passed/failed counts
    passed = 0
    failed = 0
    m = re.search(r"(\d+)\s+passed", out)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", out)
    if m:
        failed = int(m.group(1))

    # Truncate log if huge
    max_output = 20000
    if len(out) > max_output:
        half = max_output // 2
        log = out[:half] + "\n…(truncated)…\n" + out[-half:]
    else:
        log = out

    return {
        "exit_code": exit_code,
        "passed": passed,
        "failed": failed,
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