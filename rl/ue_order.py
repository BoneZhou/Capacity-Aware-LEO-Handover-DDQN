"""Seeded per-slot UE ordering shared by policy export and evaluation."""

import numpy as np


def generate_ue_orders(num_users: int, num_slots: int, seed: int) -> np.ndarray:
    """Return one permutation without replacement for every time slot."""
    if num_users <= 0:
        raise ValueError(f"num_users must be positive, got {num_users}")
    if num_slots <= 0:
        raise ValueError(f"num_slots must be positive, got {num_slots}")

    rng = np.random.default_rng(seed)
    orders = np.empty((num_slots, num_users), dtype=np.int64)
    for t in range(num_slots):
        orders[t] = rng.permutation(num_users)
    return orders
