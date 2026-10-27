import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

import Dueling_ddqn as base
import run_artifacts as artifacts


EXPERT_RATIO_INITIAL = 0.10
EXPERT_RATIO_DECAY_EPISODES = 100
BaseReplay = base.Replay


def expert_ratio_at_episode(episode_index: int) -> float:
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return EXPERT_RATIO_INITIAL * max(
        0.0, 1.0 - episode_index / EXPERT_RATIO_DECAY_EPISODES
    )


def stochastic_sample_count(batch_size: int, ratio: float) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    expected = batch_size * ratio
    count = math.floor(expected)
    fractional = expected - count
    if fractional > 0.0 and np.random.random() < fractional:
        count += 1
    return count


class ScheduledExpertReplay:
    """Static expert replay plus a FIFO online replay with scheduled mixing."""

    def __init__(
        self,
        online_capacity: int,
        *,
        pretraining_updates: int,
        transitions_per_episode: int,
    ):
        if pretraining_updates <= 0:
            raise ValueError("Scheduled expert replay requires positive pretraining_updates")
        if transitions_per_episode <= 0:
            raise ValueError("transitions_per_episode must be positive")
        self.online = BaseReplay(online_capacity)
        self.expert = []
        self.pretraining_updates = int(pretraining_updates)
        self.transitions_per_episode = int(transitions_per_episode)
        self.pretraining_samples = 0
        self.main_training = False
        self.online_pushes = 0
        self.episode_sample_counts = defaultdict(lambda: {"expert": 0, "online": 0})
        self.last_sample_counts = {"expert": 0, "online": 0}

    @property
    def current_episode(self) -> int:
        return self.online_pushes // self.transitions_per_episode

    def push(self, *args):
        transition = base.Transition(*args)
        if not self.main_training:
            self.expert.append(transition)
            return
        self.online.push(*transition)
        self.online_pushes += 1

    def _sample_expert_items(self, count: int):
        if count == 0:
            return []
        if len(self.expert) < count:
            raise ValueError(
                f"Expert buffer has {len(self.expert)} items, cannot sample {count}"
            )
        indices = np.random.choice(len(self.expert), count, replace=False)
        return [self.expert[int(index)] for index in indices]

    def _sample_online_items(self, count: int):
        if count == 0:
            return []
        if len(self.online) < count:
            raise ValueError(
                f"Online replay has {len(self.online)} items, cannot sample {count}"
            )
        indices = np.random.choice(len(self.online), count, replace=False)
        return [self.online[int(index)] for index in indices]

    def sample(self, batch_size: int):
        if not self.main_training:
            items = self._sample_expert_items(batch_size)
            self.pretraining_samples += 1
            if self.pretraining_samples == self.pretraining_updates:
                self.main_training = True
            self.last_sample_counts = {"expert": batch_size, "online": 0}
            return base.Transition(*zip(*items))

        # Keep the original online warm-up: callers see only len(online).
        if len(self.online) < batch_size:
            raise ValueError("Online replay has not reached the batch-size warm-up")

        episode = self.current_episode
        ratio = expert_ratio_at_episode(episode)
        expert_count = stochastic_sample_count(batch_size, ratio)
        online_count = batch_size - expert_count
        items = self._sample_expert_items(expert_count)
        items.extend(self._sample_online_items(online_count))
        np.random.shuffle(items)

        counts = self.episode_sample_counts[episode]
        counts["expert"] += expert_count
        counts["online"] += online_count
        self.last_sample_counts = {
            "expert": expert_count,
            "online": online_count,
        }
        return base.Transition(*zip(*items))

    def __len__(self):
        return len(self.online) if self.main_training else len(self.expert)


def _write_artifacts_with_decay(metadata, episode_logs, paths, replay):
    total_expert = 0
    total_online = 0
    for row in episode_logs:
        episode_index = int(row["episode_index"]) - 1
        counts = replay.episode_sample_counts.get(
            episode_index, {"expert": 0, "online": 0}
        )
        expert_count = int(counts["expert"])
        online_count = int(counts["online"])
        total = expert_count + online_count
        row["scheduled_expert_ratio"] = expert_ratio_at_episode(episode_index)
        row["expert_samples"] = expert_count
        row["online_samples"] = online_count
        row["actual_expert_ratio"] = expert_count / total if total else 0.0
        total_expert += expert_count
        total_online += online_count

    metadata["program_source_path"] = os.path.abspath(__file__)
    metadata["program_source_sha256"] = artifacts.sha256_file(__file__)
    metadata["base_program_source_path"] = os.path.abspath(base.__file__)
    metadata["base_program_source_sha256"] = artifacts.sha256_file(base.__file__)
    metadata["expert_replay_schedule"] = {
        "buffer_type": "independent_static_expert_plus_fifo_online",
        "expert_buffer_size": len(replay.expert),
        "online_replay_capacity": replay.online.cap,
        "initial_ratio": EXPERT_RATIO_INITIAL,
        "decay_start_episode_index": 0,
        "decay_end_episode_index": EXPERT_RATIO_DECAY_EPISODES,
        "formula": "0.1 * max(0, 1 - episode_index / 100)",
        "episode_index_base": 0,
        "sample_count_rounding": "stochastic",
        "online_warmup_preserved": True,
        "pretraining_source": "expert_only",
    }
    metadata["expert_replay_stats"]["expert_buffer_size_final"] = len(replay.expert)
    metadata["counters"]["main_training_expert_samples"] = total_expert
    metadata["counters"]["main_training_online_samples"] = total_online

    fields = artifacts.TRAINING_LOG_FIELDS + [
        "scheduled_expert_ratio",
        "expert_samples",
        "online_samples",
        "actual_expert_ratio",
    ]
    with open(paths["metadata_path"], "x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    with open(paths["training_log_path"], "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episode_logs)


def train(args):
    if args.pretrain_updates != 2000:
        raise ValueError("This training variant requires --pretrain_updates 2000")

    replay_holder = {}
    original_replay = base.Replay
    original_writer = base.write_run_artifacts

    def replay_factory(capacity):
        replay = ScheduledExpertReplay(
            capacity,
            pretraining_updates=args.pretrain_updates,
            transitions_per_episode=_scenario_transition_count(args.input),
        )
        replay_holder["replay"] = replay
        return replay

    def artifact_writer(metadata, episode_logs, paths):
        _write_artifacts_with_decay(
            metadata, episode_logs, paths, replay_holder["replay"]
        )

    base.Replay = replay_factory
    base.write_run_artifacts = artifact_writer
    try:
        return base.train(args)
    finally:
        base.Replay = original_replay
        base.write_run_artifacts = original_writer


def _scenario_transition_count(input_path: str) -> int:
    with open(input_path, "r", encoding="utf-8") as file:
        header = [int(value) for value in file.readline().replace(",", " ").split()]
    if len(header) < 4:
        raise ValueError(f"Bad input header: expected n m Tb Te, got {header}")
    _, ue_count, begin, end = header[:4]
    if ue_count <= 0 or end <= begin:
        raise ValueError(f"Bad scenario dimensions: m={ue_count}, Tb={begin}, Te={end}")
    return ue_count * (end - begin)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input_angle", type=str, required=True)
    parser.add_argument("--mshbo_csv", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_capacity", type=int, default=200000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=30000)
    parser.add_argument("--target_update", type=int, default=2000)
    parser.add_argument("--lambda_handover", type=float, default=3.0)
    parser.add_argument("--lambda_block", type=float, default=12.0)
    parser.add_argument("--lambda_throughput", type=float, default=2.0)
    parser.add_argument("--stay_bonus", type=float, default=0.05)
    parser.add_argument("--lookahead", type=int, default=5)
    parser.add_argument("--lookahead_penalty", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--sticky_prob", type=float, default=0.2)
    parser.add_argument("--pretrain_updates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_order_seed", type=int, default=None)
    parser.add_argument("--demo_order_seed", type=int, default=None)
    parser.add_argument("--eval_order_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_out", type=str, default=None)
    parser.add_argument("--policy_out", type=str, default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
