"""Dependency-free access to the repository's authoritative JSON configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / name
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported or missing schema_version in {path}")
    return value


def simulation_config() -> dict[str, Any]:
    return load_config("simulation_config.json")


def experiment_config() -> dict[str, Any]:
    return load_config("experiment_scenarios.json")


def method_config() -> dict[str, Any]:
    return load_config("method_config.json")


def reward_config_for(method: str) -> dict[str, float | int]:
    methods = load_config("reward_config.json")["methods"]
    if method not in methods:
        raise KeyError(f"Unknown reward configuration: {method}")
    return dict(methods[method])


def apply_training_defaults(parser: argparse.ArgumentParser, method: str) -> None:
    """Apply JSON defaults; explicit CLI arguments continue to take precedence."""
    reward = reward_config_for(method)
    strategy = method_config()["rl"]
    parser.set_defaults(
        lambda_handover=reward["lambda_handover"],
        lambda_block=reward["lambda_block"],
        lambda_throughput=reward["lambda_throughput"],
        stay_bonus=reward["stay_bonus"],
        lookahead=reward["lookahead"],
        lookahead_penalty=reward["lookahead_penalty"],
        topk=strategy["top_k"],
        sticky_prob=strategy["sticky_probability"],
        seed=strategy["training_seed"],
        eval_order_seed=strategy["evaluation_order_seed"],
    )
    if method in {"proposed", "proposed_no_capacity_state"}:
        parser.set_defaults(pretrain_updates=strategy["expert_pretrain_updates"])
