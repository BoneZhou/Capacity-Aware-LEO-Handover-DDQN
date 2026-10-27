"""Shared validation and dispatch for the public experiment entry points."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "rl"
CONFIGS = ROOT / "configs"

CHECKPOINTS = {
    "Proposed": RL / "runs/proposed_persistent_topk_seed42_300ep/checkpoints/dqn_imi.pt",
    "Dueling DDQN (No-MSHBO)": RL / "runs/no_mshbo_persistent_topk_seed42_300ep/checkpoints/dqn_no_mshbo.pt",
    "DDQN": RL / "runs/ddqn_persistent_topk_seed42_300ep/standard_ddqn_no_mshbo.pt",
    "Proposed w/o Remaining-Capacity State": (
        RL / "runs/no_capacity_state_persistent_topk_seed42_300ep/checkpoints/dqn_imi_no_capacity_state.pt"
    ),
}


def load_json(name: str) -> dict:
    with (CONFIGS / name).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_inputs(methods: list[str]) -> None:
    required = [ROOT / "input.txt", ROOT / "input_angle.txt"]
    missing_inputs = [path for path in required if not path.is_file()]
    if missing_inputs:
        raise SystemExit("Missing simulation input(s): " + ", ".join(map(str, missing_inputs)))
    missing_checkpoints = [f"{method}: {CHECKPOINTS[method]}" for method in methods if not CHECKPOINTS[method].is_file()]
    if missing_checkpoints:
        message = [
            "RL checkpoint validation failed.",
            "This public repository intentionally does not include trained weights.",
            "Train the required models with the commands in README.md, then place or copy the resulting weights at:",
            *missing_checkpoints,
        ]
        raise SystemExit("\n".join(message))


def run_evaluator(filename: str) -> None:
    completed = subprocess.run([sys.executable, str(RL / filename)], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
