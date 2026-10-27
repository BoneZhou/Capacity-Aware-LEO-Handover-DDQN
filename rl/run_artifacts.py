"""Safe run-directory, hashing, metadata, and episode-log helpers."""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _unique_default_output_dir(model_slug, seed):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.abspath(os.path.join("runs", f"{model_slug}_seed{seed}_{stamp}"))
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def prepare_run_paths(
    args,
    model_slug,
    checkpoint_filename,
    policy_filename,
):
    """Resolve a self-contained run directory and refuse destructive reuse."""
    requested_output_dir = getattr(args, "output_dir", None)
    output_dir = (
        os.path.abspath(requested_output_dir)
        if requested_output_dir
        else _unique_default_output_dir(model_slug, int(args.seed))
    )

    requested_checkpoint = getattr(args, "checkpoint_out", None)
    requested_policy = getattr(args, "policy_out", None)
    checkpoint_path = os.path.abspath(
        requested_checkpoint
        if requested_checkpoint
        else os.path.join(output_dir, "checkpoints", checkpoint_filename)
    )
    policy_path = os.path.abspath(
        requested_policy
        if requested_policy
        else os.path.join(output_dir, policy_filename)
    )
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    training_log_path = os.path.join(output_dir, "training_log.csv")

    protected = [checkpoint_path, policy_path, metadata_path, training_log_path]
    existing = [path for path in protected if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing run artifacts; choose a new --output_dir "
            f"and/or output paths: {existing}"
        )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    return {
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "policy_path": policy_path,
        "metadata_path": metadata_path,
        "training_log_path": training_log_path,
    }


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_time_first_metadata(
    *,
    model_variant,
    args,
    env,
    paths,
    training_order_seed,
    demonstration_order_seed,
    evaluation_order_seed,
    demonstration_enabled,
    actual_pretraining_updates,
    best_episode,
    best_episode_return,
    counters,
    source_path,
    rate_normalization_max_mbps,
    action_to_rate_id,
    started_at_utc,
    device,
    expert_replay_stats=None,
):
    expert_path = getattr(args, "mshbo_csv", None)
    if expert_path and os.path.exists(expert_path):
        expert_status = "loaded"
        expert_hash = sha256_file(expert_path)
        expert_abs_path = os.path.abspath(expert_path)
    elif expert_path:
        expert_status = "unavailable_file_not_found"
        expert_hash = None
        expert_abs_path = os.path.abspath(expert_path)
    else:
        expert_status = "disabled"
        expert_hash = None
        expert_abs_path = None

    ue_order_path = os.path.join(os.path.dirname(os.path.abspath(source_path)), "ue_order.py")
    return {
        "model_variant": model_variant,
        "training_mode": "time_first",
        "state_dimension": int(env.state_dim),
        "action_dimension": int(env.n_actions),
        "mshbo_demonstration_enabled": bool(demonstration_enabled),
        "checkpoint_format": "online_network_state_dict_only",
        "training_seed": int(args.seed),
        "training_order_seed": int(training_order_seed),
        "demonstration_order_seed": (
            int(demonstration_order_seed) if demonstration_order_seed is not None else None
        ),
        "evaluation_order_seed": int(evaluation_order_seed),
        "episodes": int(args.episodes),
        "best_episode": int(best_episode) if best_episode is not None else None,
        "best_episode_return": float(best_episode_return),
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "gamma": float(args.gamma),
        "epsilon_start": float(args.eps_start),
        "epsilon_end": float(args.eps_end),
        "epsilon_decay_slots": int(args.eps_decay),
        "target_network_update_slots": int(args.target_update),
        "replay_buffer_capacity": int(args.replay_capacity),
        "pretraining_updates_requested": int(getattr(args, "pretrain_updates", 0)),
        "pretraining_updates_actual": int(actual_pretraining_updates),
        "training_loss": "unavailable_existing_ddqn_update_returns_success_boolean_only",
        "top_k": int(args.topk),
        "td_target_mask_type": "next_state_top_k",
        "policy_export_mask_type": "top_k",
        "capacity_accounting": "connection_persistent",
        "block_action_always_legal": True,
        "sticky_probability": float(args.sticky_prob),
        "lookahead_slots": int(args.lookahead),
        "reward_weights": {
            "handover": float(args.lambda_handover),
            "blocking": float(args.lambda_block),
            "throughput": float(args.lambda_throughput),
            "stay_bonus": float(args.stay_bonus),
            "lookahead_penalty": float(args.lookahead_penalty),
        },
        "throughput_normalization_max_mbps": float(rate_normalization_max_mbps),
        "ue_count": int(env.m),
        "satellite_count": int(env.n),
        "time_slot_count": int(env.T),
        "time_interval_convention": "[Tb, Te)",
        "satellite_capacity": [int(value) for value in env.cap.tolist()],
        "satellite_id_mapping": [int(value) for value in action_to_rate_id.tolist()],
        "input_path": os.path.abspath(args.input),
        "input_data_sha256": sha256_file(args.input),
        "rate_table_path": os.path.abspath(args.input_angle),
        "rate_table_sha256": sha256_file(args.input_angle),
        "mshbo_expert_csv_path": expert_abs_path,
        "mshbo_expert_csv_sha256": expert_hash,
        "mshbo_expert_status": expert_status,
        "expert_replay_stats": _json_safe(expert_replay_stats),
        "ue_order_rng": "numpy.random.default_rng",
        "ue_order_granularity": "one_permutation_per_time_slot",
        "output_directory": paths["output_dir"],
        "checkpoint_path": paths["checkpoint_path"],
        "checkpoint_file_name": os.path.basename(paths["checkpoint_path"]),
        "policy_export_path": paths["policy_path"],
        "policy_export_file_name": os.path.basename(paths["policy_path"]),
        "training_log_path": paths["training_log_path"],
        "metadata_path": paths["metadata_path"],
        "program_source_path": os.path.abspath(source_path),
        "program_source_sha256": sha256_file(source_path),
        "ue_order_source_path": ue_order_path,
        "ue_order_source_sha256": sha256_file(ue_order_path),
        "execution_started_at_utc": started_at_utc,
        "execution_completed_at_utc": utc_now_iso(),
        "device": str(device),
        "counters": _json_safe(counters),
        "cli_arguments": _json_safe(vars(args)),
    }


TRAINING_LOG_FIELDS = [
    "episode_index",
    "episode_return",
    "best_return_so_far",
    "best_checkpoint_updated",
    "epsilon_first_slot",
    "epsilon_last_slot",
    "decisions",
    "transitions_completed",
    "blocks",
    "handovers",
    "gradient_updates",
]


def write_run_artifacts(metadata, episode_logs, paths):
    with open(paths["metadata_path"], "x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(paths["training_log_path"], "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(episode_logs)

