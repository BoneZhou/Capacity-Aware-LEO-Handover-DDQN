import argparse
import os
import csv
import math
import random
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ue_order import generate_ue_orders
from run_artifacts import (
    build_time_first_metadata,
    prepare_run_paths,
    utc_now_iso,
    write_run_artifacts,
)
from config_loader import apply_training_defaults, reward_config_for


# Legacy achievable-rate data is complete only for t=0..3598.  input.txt stores
# Te=3599 as the exclusive upper boundary for this deliberately truncated run.
RATE_NORMALIZATION_MAX_MBPS = reward_config_for("standard_ddqn")["rate_normalization_max_mbps"]
DEFAULT_CHECKPOINT_PATH = "./checkpoints/standard_ddqn_no_mshbo.pt"
DEFAULT_POLICY_PATH = "policy_actions_standard_ddqn_no_mshbo_251.csv"
INPUT_ACTION_TO_RATE_ID = np.array(
    [
        36, 27, 18, 10, 2, 44, 35, 26, 17, 9,
        1, 43, 34, 25, 16, 8, 0, 42, 33, 24,
        15, 7, 49, 41, 32, 23, 14, 6, 48, 40,
        31, 22, 13, 5, 47, 39, 30, 21, 12, 4,
        46, 38, 29, 3, 11, 19, 20, 28, 37, 45,
    ],
    dtype=np.int64,
)

# -------------------- Replay buffers --------------------
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done", "mask"))
Pending = namedtuple("Pending", ("state", "action", "reward"))


def seed_everything(seed: int):
    """Seed every RNG used by training, within framework/hardware guarantees."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def epsilon_at_slot(eps_start: float, eps_end: float, eps_decay: int, slot_step: int) -> float:
    """Exponential epsilon schedule indexed by completed global time slots."""
    decay = max(1, eps_decay)
    return eps_end + (eps_start - eps_end) * math.exp(-slot_step / decay)


class Replay:
    def __init__(self, cap: int = 200000):
        if cap <= 0:
            raise ValueError(f"Replay capacity must be positive, got {cap}")
        self.cap = cap
        self._storage = [None] * cap
        self._start = 0
        self._size = 0

    def push(self, *args):
        write_index = (self._start + self._size) % self.cap
        self._storage[write_index] = Transition(*args)
        if self._size < self.cap:
            self._size += 1
        else:
            self._start = (self._start + 1) % self.cap

    def __getitem__(self, logical_index: int):
        """Return an item by deque-compatible chronological index."""
        if not isinstance(logical_index, (int, np.integer)):
            raise TypeError("Replay indices must be integers")
        logical_index = int(logical_index)
        if logical_index < 0:
            logical_index += self._size
        if not 0 <= logical_index < self._size:
            raise IndexError("Replay index out of range")
        physical_index = (self._start + logical_index) % self.cap
        return self._storage[physical_index]

    def sample(self, batch: int):
        idx = np.random.choice(len(self), batch, replace=False)
        items = [self[i] for i in idx]
        return Transition(*zip(*items))

    def __len__(self):
        return self._size


# -------------------- Standard Double DQN --------------------
class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.feat = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(nn.Linear(256, n_actions))

    def forward(self, x):
        single = False
        if x.dim() == 1:  # [in_dim] -> [1, in_dim]
            x = x.unsqueeze(0)
            single = True

        h = self.feat(x)
        q = self.q_head(h)  # [B, n_actions]
        return q.squeeze(0) if single else q


# -------------------- Environment --------------------
class HandoverEnv:
    """
    Input file format:
      First line: n m T_b T_e
      Next n lines: begin end capacity

    An episode processes every UE once per slot in a seeded random order.
    Active connections reserve capacity across slots; updates are shared immediately.

    Action space:
      0 ~ n-1 : choose satellite
      n       : BLOCK action
    """

    def __init__(
        self,
        path: str,
        rate_path: str,
        lambda_h: float = 3.0,
        lambda_b: float = 12.0,
        lambda_t: float = 2.0,
        stay_bonus: float = 0.05,
        lookahead: int = 5,
        lookahead_penalty: float = 0.1,
        seed: int = 42,
        training_order_seed: int = None,
        demonstration_order_seed: int = None,
        evaluation_order_seed: int = 42,
    ):
        self.path = path
        self.rate_path = rate_path
        self.rng = random.Random(seed)
        self.training_order_seed = seed if training_order_seed is None else training_order_seed
        self.demonstration_order_seed = (
            seed if demonstration_order_seed is None else demonstration_order_seed
        )
        self.evaluation_order_seed = evaluation_order_seed
        self.training_order_rng = np.random.default_rng(self.training_order_seed)
        self.demo_order_rng = np.random.default_rng(self.demonstration_order_seed)
        self.evaluation_order_rng = np.random.default_rng(self.evaluation_order_seed)
        self.order_rng = self.training_order_rng

        self.lambda_h = lambda_h
        self.lambda_b = lambda_b
        self.lambda_t = lambda_t
        self.stay_bonus = stay_bonus
        self.lookahead = lookahead
        self.lookahead_penalty = lookahead_penalty

        self._load()
        self._load_rate()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            line = f.readline()
            nums = [int(x) for x in line.replace(",", " ").split()]
            if len(nums) < 4:
                raise ValueError(f"Bad header: need 4 ints, got {nums}")

            self.n, self.m, self.Tb, self.Te = nums[:4]

            self.begin = np.zeros(self.n, dtype=int)
            self.end = np.zeros(self.n, dtype=int)
            self.cap = np.zeros(self.n, dtype=int)

            for i in range(self.n):
                b, e, c = map(int, f.readline().strip().split())
                self.begin[i] = b
                self.end[i] = e
                self.cap[i] = c

        # Te is an exclusive boundary.  The legacy experiment is deliberately
        # truncated to t=0..3598 because no trustworthy t=3599 rate exists.
        self.T = self.Te - self.Tb
        if self.T <= 0:
            raise ValueError(f"Bad time interval: expected Tb < Te, got {self.Tb}, {self.Te}")

        # Visibility V[s, t]
        self.V = np.zeros((self.n, self.T), dtype=np.uint8)
        for s in range(self.n):
            b = max(self.begin[s], self.Tb)
            e = min(self.end[s], self.Te)
            if e > b:
                self.V[s, b - self.Tb : e - self.Tb] = 1

        self.block_action = self.n
        self.n_actions = self.n + 1
        self.state_dim = self.n * 5 + 1  # vis, rem, last, next_vis, rate, t_ratio

    def _load_rate(self):
        """Load, validate, reorder, and normalize rate[satellite, local_time]."""
        if not self.rate_path:
            raise ValueError("input_angle is required for achievable-rate state and reward")
        if self.n != len(INPUT_ACTION_TO_RATE_ID):
            raise ValueError(
                f"Fixed legacy rate mapping requires {len(INPUT_ACTION_TO_RATE_ID)} satellites, got {self.n}"
            )
        if sorted(INPUT_ACTION_TO_RATE_ID.tolist()) != list(range(self.n)):
            raise ValueError("Legacy action-to-rate mapping is not a complete bijection")

        original_rate = np.zeros((self.n, self.T), dtype=np.float32)

        seen = np.zeros((self.n, self.T), dtype=np.uint8)

        with open(self.rate_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(
                        f"Bad rate row at line {line_no}: need 'sat time rate_mbps', got {line.strip()!r}"
                    )

                try:
                    sat = int(parts[0])
                    absolute_t = int(parts[1])
                    rate_mbps = float(parts[2])
                except ValueError as exc:
                    raise ValueError(
                        f"Bad rate values at line {line_no}: {line.strip()!r}"
                    ) from exc

                local_t = absolute_t - self.Tb
                if not 0 <= sat < self.n:
                    raise ValueError(
                        f"Rate satellite id out of range at line {line_no}: {sat}, expected 0..{self.n - 1}"
                    )
                if not 0 <= local_t < self.T:
                    raise ValueError(
                        f"Rate time out of range at line {line_no}: {absolute_t}, "
                        f"expected {self.Tb}..{self.Te - 1}"
                    )
                if not math.isfinite(rate_mbps) or rate_mbps < 0.0:
                    raise ValueError(
                        f"Rate must be finite and non-negative at line {line_no}: {rate_mbps}"
                    )
                if seen[sat, local_t]:
                    raise ValueError(
                        f"Duplicate rate entry at line {line_no}: satellite={sat}, time={absolute_t}"
                    )

                original_rate[sat, local_t] = rate_mbps
                seen[sat, local_t] = 1

        missing = np.argwhere(seen == 0)
        if len(missing) > 0:
            preview = [
                (int(sat), int(local_t + self.Tb))
                for sat, local_t in missing[:10]
            ]
            raise ValueError(
                f"Incomplete rate table: missing {len(missing)} of {self.n * self.T} entries; "
                f"first missing (satellite, time) pairs: {preview}"
            )

        self.rate_mapping = INPUT_ACTION_TO_RATE_ID.copy()
        self.rate = original_rate[self.rate_mapping, :].copy()
        self.rate_norm = self.rate / np.float32(RATE_NORMALIZATION_MAX_MBPS)

        tolerance = 1e-6
        if not math.isclose(
            float(self.rate.max()), RATE_NORMALIZATION_MAX_MBPS, rel_tol=0.0, abs_tol=1e-3
        ):
            raise ValueError(
                f"Unexpected dataset maximum rate: {self.rate.max()} Mbps; "
                f"expected {RATE_NORMALIZATION_MAX_MBPS} Mbps"
            )
        if np.any(self.rate_norm < -tolerance) or np.any(self.rate_norm > 1.0 + tolerance):
            raise ValueError(
                f"Normalized rate outside [0,1]: min={self.rate_norm.min()}, "
                f"max={self.rate_norm.max()}"
            )
        if np.any((self.rate == 0.0) & (self.rate_norm != 0.0)):
            raise ValueError("Zero-rate entries did not remain zero after normalization")

        support_mismatch = np.argwhere(self.V != (self.rate > 0.0))
        if len(support_mismatch) > 0:
            preview = [(int(s), int(t + self.Tb)) for s, t in support_mismatch[:10]]
            raise ValueError(
                f"Visibility/rate support mismatch: {len(support_mismatch)} entries; "
                f"first mismatches: {preview}"
            )

    def reset_multi_user_episode(self):
        """Reset shared occupancy and every UE's connection history."""
        self.used = np.zeros((self.n, self.T), dtype=np.int16)
        self.occupancy = np.zeros(self.n, dtype=np.int16)
        self._active_slot = -1
        self.last_sat = np.full(self.m, -1, dtype=np.int64)
        self.last_effective_sat = np.full(self.m, -1, dtype=np.int64)
        self.last_reward_components = [None for _ in range(self.m)]

    def _assert_capacity_consistency(self, t: int):
        expected = np.bincount(
            self.last_sat[self.last_sat >= 0], minlength=self.n
        ).astype(np.int16)
        if not np.array_equal(self.occupancy, expected):
            raise AssertionError("Occupancy does not match active UE-satellite associations")
        if np.any(self.occupancy < 0) or np.any(self.occupancy > self.cap):
            raise AssertionError("Satellite occupancy is outside [0, capacity]")
        if not np.array_equal(self.used[:, t], self.occupancy):
            raise AssertionError("Current-slot occupancy snapshot is inconsistent")

    def _prepare_slot(self, t: int):
        """Carry active connections forward and release those no longer visible."""
        if t == self._active_slot:
            return
        if t < self._active_slot:
            raise ValueError(f"Slots must be processed in order: active={self._active_slot}, requested={t}")
        for user, satellite in enumerate(self.last_sat):
            satellite = int(satellite)
            if satellite >= 0 and self.V[satellite, t] == 0:
                self.occupancy[satellite] -= 1
                self.last_sat[user] = -1
                self.last_effective_sat[user] = -1
        self._active_slot = t
        self.used[:, t] = self.occupancy
        self._assert_capacity_consistency(t)

    def next_user_order(self):
        """Backward-compatible alias for the formal-training order stream."""
        return self.order_rng.permutation(self.m)

    def next_training_user_order(self):
        rng = getattr(self, "training_order_rng", self.order_rng)
        return rng.permutation(self.m)

    def next_demo_user_order(self):
        rng = getattr(self, "demo_order_rng", self.order_rng)
        return rng.permutation(self.m)

    def reset_evaluation_order_rng(self, seed=None):
        if seed is not None:
            self.evaluation_order_seed = seed
        self.evaluation_order_rng = np.random.default_rng(self.evaluation_order_seed)

    def next_evaluation_user_order(self):
        return self.evaluation_order_rng.permutation(self.m)

    def _validate_turn(self, user: int, t: int):
        if not 0 <= user < self.m:
            raise IndexError(f"User index out of range: {user}, expected 0..{self.m - 1}")
        if not 0 <= t < self.T:
            raise IndexError(f"Time index out of range: {t}, expected 0..{self.T - 1}")

    def make_state(self, user: int, t: int):
        self._validate_turn(user, t)
        self._prepare_slot(t)
        vis = self.V[:, t].astype(np.float32)

        rem = np.zeros(self.n, dtype=np.float32)
        for s in range(self.n):
            rem[s] = max(0.0, (self.cap[s] - self.used[s, t])) / max(1, self.cap[s])

        last = np.zeros(self.n, dtype=np.float32)
        if 0 <= self.last_sat[user] < self.n:
            last[self.last_sat[user]] = 1.0

        t_next = min(t + 1, self.T - 1)
        next_vis = self.V[:, t_next].astype(np.float32)
        rate = self.rate_norm[:, t].astype(np.float32)

        t_ratio = np.array([(t + 1) / self.T], dtype=np.float32)

        return np.concatenate([vis, rem, last, next_vis, rate, t_ratio], axis=0)

    def legal_mask(self, user: int, t: int):
        self._validate_turn(user, t)
        self._prepare_slot(t)
        # satellites + block action
        mask = np.zeros(self.n_actions, dtype=np.float32)

        # real satellites
        sat_mask = self.V[:, t].astype(np.float32)
        current_satellite = int(self.last_sat[user])
        for s in range(self.n):
            if self.occupancy[s] >= self.cap[s] and s != current_satellite:
                sat_mask[s] = 0.0

        mask[:self.n] = sat_mask

        # block action is always legal
        mask[self.block_action] = 1.0
        return mask

    def step_user(self, user: int, t: int, action: int):
        """Allocate one UE immediately within the current shared time slot."""
        self._validate_turn(user, t)
        self._prepare_slot(t)
        blocked = False
        components = {
            "throughput": 0.0,
            "blocking": 0.0,
            "handover": 0.0,
            "stay": 0.0,
            "lookahead": 0.0,
        }

        # ----- BLOCK action -----
        if action == self.block_action:
            blocked = True
            components["blocking"] = -self.lambda_b
            previous_satellite = int(self.last_sat[user])
            if previous_satellite >= 0:
                self.occupancy[previous_satellite] -= 1
            self.last_sat[user] = -1
            self.last_effective_sat[user] = -1

        # ----- invalid action -----
        elif action < 0 or action >= self.n:
            blocked = True
            components["blocking"] = -self.lambda_b

        else:
            mask_now = self.legal_mask(user, t)

            # illegal satellite => blocked
            if mask_now[action] == 0:
                blocked = True
                components["blocking"] = -self.lambda_b
            else:
                prev_sat = self.last_sat[user]
                if action != prev_sat:
                    # Acquire first; only a successful handover releases the old reservation.
                    self.occupancy[action] += 1
                    if prev_sat != -1:
                        self.occupancy[prev_sat] -= 1
                components["throughput"] = self.lambda_t * float(self.rate_norm[action, t])

                # stay / sat->sat handover reward
                if prev_sat != -1 and action == prev_sat:
                    components["stay"] = self.stay_bonus
                elif prev_sat != -1 and action != prev_sat:
                    components["handover"] = -self.lambda_h

                # future visibility penalty
                if self.lookahead > 0:
                    lose_soon = 0
                    horizon = min(t + self.lookahead, self.T - 1)
                    for tt in range(t + 1, horizon + 1):
                        if self.V[action, tt] == 0:
                            lose_soon += 1
                    if lose_soon > 0:
                        components["lookahead"] = -self.lookahead_penalty * lose_soon

                self.last_sat[user] = action
                self.last_effective_sat[user] = action

        self.used[:, t] = self.occupancy
        self._assert_capacity_consistency(t)
        reward = float(sum(components.values()))
        components["total"] = reward
        self.last_reward_components[user] = components
        return reward, blocked

# -------------------- helpers --------------------
def masked_argmax(q: torch.Tensor, mask: np.ndarray) -> int:
    q = q.clone()
    illegal = (torch.from_numpy(1 - mask).bool()).to(q.device)
    q[illegal] = -1e9
    return int(q.argmax().item())


def masked_next_argmax(q_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Batched argmax restricted to legal next-state actions."""
    if q_values.shape != masks.shape:
        raise ValueError(
            f"Q/mask shape mismatch: q_values={tuple(q_values.shape)}, masks={tuple(masks.shape)}"
        )
    if torch.any(masks.sum(dim=1) <= 0):
        raise ValueError("Every next-state legal mask must contain at least one legal action")
    masked_q = q_values.masked_fill(masks <= 0, torch.finfo(q_values.dtype).min)
    return masked_q.argmax(dim=1, keepdim=True)


def ddqn_update(q, tq, opt, buf, batch_size: int, gamma: float, device) -> bool:
    """Perform at most one masked Double-DQN update from replay."""
    if len(buf) < batch_size:
        return False

    batch = buf.sample(batch_size)
    valid_idx = [i for i, action in enumerate(batch.action) if 0 <= int(action) < q.n_actions]
    if not valid_idx:
        return False

    s_b = torch.tensor(np.stack([batch.state[i] for i in valid_idx]), dtype=torch.float32, device=device)
    a_b = torch.tensor([int(batch.action[i]) for i in valid_idx], dtype=torch.int64, device=device)
    ns_b = torch.tensor(np.stack([batch.next_state[i] for i in valid_idx]), dtype=torch.float32, device=device)
    r_b = torch.tensor([batch.reward[i] for i in valid_idx], dtype=torch.float32, device=device)
    d_b = torch.tensor([batch.done[i] for i in valid_idx], dtype=torch.float32, device=device)
    m_b = torch.tensor(np.stack([batch.mask[i] for i in valid_idx]), dtype=torch.float32, device=device)

    q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        a_star = masked_next_argmax(q(ns_b), m_b)
        tq_ns = tq(ns_b).gather(1, a_star).squeeze(1)
        target = r_b + (1.0 - d_b) * gamma * tq_ns

    loss = nn.SmoothL1Loss()(q_sa, target)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), 1.0)
    opt.step()
    return True


def topk_mask(mask, used, cap, t, last, block_action: int, K: int = 0):
    """
    Keep:
      - block action always
      - last satellite (if legal)
      - top (K-1) legal satellites by remaining capacity
    """
    new_mask = np.zeros_like(mask)
    new_mask[block_action] = 1.0  # block always legal

    sat_mask = mask[:block_action].copy()
    idx = np.where(sat_mask > 0)[0]

    if K is None or K <= 0:
        new_mask[:block_action] = sat_mask
        return new_mask

    if len(idx) <= K:
        new_mask[:block_action] = sat_mask
        return new_mask

    keep = []
    if last in idx:
        keep.append(last)

    rem = [(s, cap[s] - used[s, t]) for s in idx if s != last]
    rem.sort(key=lambda x: x[1], reverse=True)

    for s, _ in rem:
        if len(keep) >= K:
            break
        keep.append(s)

    for s in keep:
        new_mask[s] = 1.0

    return new_mask


# -------------------- training --------------------
def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    training_order_seed = (
        args.seed if getattr(args, "training_order_seed", None) is None
        else args.training_order_seed
    )
    evaluation_order_seed = args.eval_order_seed
    paths = prepare_run_paths(
        args,
        model_slug="time_first_standard_ddqn",
        checkpoint_filename="standard_ddqn_no_mshbo.pt",
        policy_filename="policy_actions_standard_ddqn_no_mshbo_251.csv",
    )
    started_at_utc = utc_now_iso()
    started_at = time.perf_counter()

    env = HandoverEnv(
        args.input,
        rate_path=args.input_angle,
        lambda_h=args.lambda_handover,
        lambda_b=args.lambda_block,
        lambda_t=args.lambda_throughput,
        stay_bonus=args.stay_bonus,
        lookahead=args.lookahead,
        lookahead_penalty=args.lookahead_penalty,
        seed=args.seed,
        training_order_seed=training_order_seed,
        demonstration_order_seed=args.seed,
        evaluation_order_seed=evaluation_order_seed,
    )

    in_dim = env.state_dim
    n_actions = env.n_actions

    q = QNet(in_dim, n_actions).to(device)
    tq = QNet(in_dim, n_actions).to(device)
    tq.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=args.lr)
    buf = Replay(args.replay_capacity)
    gradient_update_count = 0

    # This controlled baseline starts from an empty replay and has no expert-data path.
    demo_transition_count = 0
    demo_pretraining_update_count = 0
    if len(buf) != 0:
        raise RuntimeError("Standard no-MSHBO replay must be empty before environment interaction")

    print("Model: Standard DDQN")
    print("Dueling architecture: OFF")
    print("MSHBO expert: OFF")
    print("State dim:", in_dim)
    print("Seed:", args.seed)
    print("Episodes:", args.episodes)
    print("U:", env.m)
    print("T:", env.T)
    print("Epsilon schedule: slot-based")
    print("Target schedule: slot-based")
    print("Initial replay size:", len(buf))
    print("MSHBO/demo transitions loaded:", demo_transition_count)
    print("Demo pretraining gradient updates:", demo_pretraining_update_count)

    # ---------- main RL training ----------
    eps_start, eps_end = args.eps_start, args.eps_end
    eps_decay = max(1, args.eps_decay)

    global_decision_step = 0
    global_slot_step = 0
    total_decisions = 0
    total_transitions_pushed = 0
    main_gradient_update_count = 0
    regular_target_sync_count = 0
    best_score = -1e18
    best_episode = None
    episode_logs = []

    for ep in range(args.episodes):
        env.reset_multi_user_episode()
        pending = [None for _ in range(env.m)]
        ep_ret = 0.0
        decisions = 0
        completed_transitions = 0
        blocks = 0
        handovers = 0
        episode_gradient_updates_before = main_gradient_update_count
        episode_epsilon_first = None
        episode_epsilon_last = None

        for t in range(env.T):
            # Every UE in this slot uses one epsilon derived from the number of
            # globally completed slots, not from the UE-decision count.
            eps = epsilon_at_slot(eps_start, eps_end, eps_decay, global_slot_step)
            if episode_epsilon_first is None:
                episode_epsilon_first = eps
            episode_epsilon_last = eps
            user_order = env.next_training_user_order()
            for raw_user in user_order:
                user = int(raw_user)

                # State and mask observe the same latest same-slot occupancy.
                state = env.make_state(user, t)
                legal_mask = env.legal_mask(user, t)
                selection_mask = topk_mask(
                    legal_mask,
                    env.used,
                    env.cap,
                    t,
                    int(env.last_sat[user]),
                    env.block_action,
                    args.topk,
                )

                # Complete only this UE's pending transition from t-1.
                if pending[user] is not None:
                    prev = pending[user]
                    buf.push(
                        prev.state,
                        prev.action,
                        state.copy(),
                        prev.reward,
                        0.0,
                        selection_mask.copy(),
                    )
                    completed_transitions += 1
                    total_transitions_pushed += 1

                if random.random() < eps:
                    action = int(np.random.choice(np.where(selection_mask > 0)[0]))
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        q_values = q(state_tensor)
                    action = masked_argmax(q_values, selection_mask)

                # Preserve sticky behavior, but validate against the current full legal mask.
                sticky_sat = int(env.last_effective_sat[user])
                if (
                    args.sticky_prob > 0
                    and sticky_sat != -1
                    and random.random() < args.sticky_prob
                    and legal_mask[sticky_sat] > 0
                ):
                    action = sticky_sat

                reward, blocked = env.step_user(user, t, action)
                components = env.last_reward_components[user]
                if components["handover"] < 0:
                    handovers += 1
                if blocked:
                    blocks += 1

                pending[user] = Pending(state.copy(), int(action), float(reward))
                ep_ret += reward
                decisions += 1
                global_decision_step += 1
                total_decisions += 1

            # Slot-based learning schedule: at most one gradient update after
            # every UE in this time slot has completed its decision.
            if ddqn_update(q, tq, opt, buf, args.batch_size, args.gamma, device):
                gradient_update_count += 1
                main_gradient_update_count += 1

            # Epsilon and regular target synchronization share the completed
            # global-slot counter. Target sync happens after this slot's update.
            global_slot_step += 1
            if global_slot_step % args.target_update == 0:
                tq.load_state_dict(q.state_dict())
                regular_target_sync_count += 1

            if (t + 1) % 500 == 0 or (t + 1) == env.T:
                print(
                    f"Episode {ep+1}/{args.episodes} | Slot {t+1}/{env.T} | "
                    f"Replay={len(buf)} | Updates={gradient_update_count} | "
                    f"GlobalDecisions={global_decision_step} | GlobalSlots={global_slot_step} | "
                    f"EpsilonUsed={eps:.6f} | TargetSyncs={regular_target_sync_count}"
                )

        # There is no t=T observation. Flush one terminal transition per UE.
        zero_state = np.zeros(env.state_dim, dtype=np.float32)
        terminal_mask = np.ones(env.n_actions, dtype=np.float32)
        for user in range(env.m):
            prev = pending[user]
            if prev is None:
                continue
            buf.push(
                prev.state,
                prev.action,
                zero_state.copy(),
                prev.reward,
                1.0,
                terminal_mask.copy(),
            )
            completed_transitions += 1
            total_transitions_pushed += 1

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{args.episodes}  Return {ep_ret:.1f}  "
                f"decisions={decisions}  transitions={completed_transitions}  "
                f"handovers={handovers}  blocks={blocks}"
            )

        best_updated = ep_ret > best_score
        if best_updated:
            best_score = ep_ret
            best_episode = ep + 1
            torch.save(q.state_dict(), paths["checkpoint_path"])

        episode_logs.append(
            {
                "episode_index": ep + 1,
                "episode_return": float(ep_ret),
                "best_return_so_far": float(best_score),
                "best_checkpoint_updated": bool(best_updated),
                "epsilon_first_slot": float(episode_epsilon_first),
                "epsilon_last_slot": float(episode_epsilon_last),
                "decisions": int(decisions),
                "transitions_completed": int(completed_transitions),
                "blocks": int(blocks),
                "handovers": int(handovers),
                "gradient_updates": int(
                    main_gradient_update_count - episode_gradient_updates_before
                ),
            }
        )

    print("Training done. Best return:", best_score)
    print("Total UE decisions:", total_decisions)
    print("Global slots:", global_slot_step)
    print("Transitions pushed:", total_transitions_pushed)
    print("Gradient updates (main training):", main_gradient_update_count)
    print("MSHBO/demo transitions loaded:", demo_transition_count)
    print("Demo pretraining gradient updates:", demo_pretraining_update_count)
    print("Gradient updates (total):", gradient_update_count)
    print("Regular target synchronizations:", regular_target_sync_count)
    print(
        "Epsilon for next slot:",
        epsilon_at_slot(eps_start, eps_end, eps_decay, global_slot_step),
    )

    # export greedy policy over all users
    q.load_state_dict(torch.load(paths["checkpoint_path"], map_location=device, weights_only=False))

    def policy_fn(s, mask):
        st = torch.tensor(s, dtype=torch.float32, device=device)
        with torch.no_grad():
            qv = q(st)
        return masked_argmax(qv, mask)

    export_policy(
        env,
        policy_fn,
        out_csv=paths["policy_path"],
        order_seed=evaluation_order_seed,
        topk=args.topk,
    )
    counters = {
        "ue_decisions": total_decisions,
        "global_slots": global_slot_step,
        "transitions_pushed": total_transitions_pushed,
        "main_gradient_updates": main_gradient_update_count,
        "demonstration_gradient_updates": 0,
        "total_gradient_updates": gradient_update_count,
        "target_synchronizations": regular_target_sync_count,
    }
    metadata = build_time_first_metadata(
        model_variant="Standard DDQN",
        args=args,
        env=env,
        paths=paths,
        training_order_seed=training_order_seed,
        demonstration_order_seed=None,
        evaluation_order_seed=evaluation_order_seed,
        demonstration_enabled=False,
        actual_pretraining_updates=0,
        best_episode=best_episode,
        best_episode_return=best_score,
        counters=counters,
        source_path=__file__,
        rate_normalization_max_mbps=RATE_NORMALIZATION_MAX_MBPS,
        action_to_rate_id=env.rate_mapping,
        started_at_utc=started_at_utc,
        device=device,
    )
    metadata["elapsed_seconds"] = time.perf_counter() - started_at
    write_run_artifacts(metadata, episode_logs, paths)
    print("Saved checkpoint:", paths["checkpoint_path"])
    print("Saved policy:", paths["policy_path"])
    print("Saved metadata:", paths["metadata_path"])
    print("Saved training log:", paths["training_log_path"])
    return metadata


def export_policy(env, policy_fn, out_csv=DEFAULT_POLICY_PATH, order_seed=42, topk=3):
    acts = []
    env.reset_multi_user_episode()
    env.reset_evaluation_order_rng(order_seed)

    for t in range(env.T):
        for raw_user in env.next_evaluation_user_order():
            user = int(raw_user)
            state = env.make_state(user, t)
            legal_mask = env.legal_mask(user, t)
            mask = topk_mask(
                legal_mask,
                env.used,
                env.cap,
                t,
                int(env.last_sat[user]),
                env.block_action,
                topk,
            )
            action = policy_fn(state, mask)

            # save as sat id or -1
            sat_out = -1 if action == env.block_action else action
            acts.append((t + env.Tb, user, sat_out))

            env.step_user(user, t, action)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "user", "sat"])
        w.writerows(acts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument(
        "--input_angle",
        type=str,
        required=True,
        help="required achievable-rate table with rows: satellite time rate_mbps",
    )
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--replay_capacity", type=int, default=200000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay", type=int, default=30000)
    ap.add_argument("--target_update", type=int, default=2000)
    ap.add_argument("--lambda_handover", type=float, default=3.0)
    ap.add_argument("--lambda_block", type=float, default=12.0)
    ap.add_argument("--lambda_throughput", type=float, default=2.0)
    ap.add_argument("--stay_bonus", type=float, default=0.05)
    ap.add_argument("--lookahead", type=int, default=5)
    ap.add_argument("--lookahead_penalty", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=3, help="keep last + top (K-1) remaining-capacity sats; 0=disable")
    ap.add_argument("--sticky_prob", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--training_order_seed",
        type=int,
        default=None,
        help="formal-training per-slot UE-order seed; defaults to --seed",
    )
    ap.add_argument(
        "--eval_order_seed",
        type=int,
        default=42,
        help="seed for the fixed per-slot UE permutations used by policy export/evaluation",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="isolated run directory; defaults to a unique runs/ directory",
    )
    ap.add_argument(
        "--checkpoint_out",
        type=str,
        default=None,
        help="optional explicit checkpoint path inside a new run",
    )
    ap.add_argument(
        "--policy_out",
        type=str,
        default=None,
        help="optional explicit policy CSV path inside a new run",
    )

    apply_training_defaults(ap, "standard_ddqn")
    args = ap.parse_args()
    train(args)
