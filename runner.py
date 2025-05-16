#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

IMAGE  = "swe-lite-runner"
MIRROR = Path.home() / ".cache" / "swe-lite" / "mirror"

def run_instance(iid: str):
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{MIRROR}:/mirror",     # still mount the git‐mirror cache
        IMAGE,
        iid
    ]
    print("🔸 Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <instance_id> …")
        sys.exit(1)

    # ensure mirror cache dir exists on the host
    MIRROR.mkdir(parents=True, exist_ok=True)
    print("🔹 Using mirror cache:", MIRROR)
    for iid in sys.argv[1:]:
        run_instance(iid)