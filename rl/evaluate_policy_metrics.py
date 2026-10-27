
import argparse, csv
import numpy as np

from ue_order import generate_ue_orders

def load_env(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        head = f.readline().strip().replace(',', ' ').split()
        if len(head) < 4:
            raise ValueError(f"Bad header in %s: need 4 ints 'n m T_b T_e', got %s" % (input_path, head))
        n, m, Tb, Te = map(int, head[:4])
        begin = np.zeros(n, dtype=int)
        end   = np.zeros(n, dtype=int)
        cap   = np.zeros(n, dtype=int)
        for i in range(n):
            b,e,c = map(int, f.readline().split())
            begin[i]=b; end[i]=e; cap[i]=c
    # Te is the exclusive upper boundary; legacy rate data supports t=0..3598.
    T = Te - Tb
    V = np.zeros((n, T), dtype=np.uint8)
    for s in range(n):
        b = max(begin[s], Tb)
        e = min(end[s], Te)
        if e > b:
            V[s, b-Tb:e-Tb] = 1
    return n, m, Tb, Te, T, cap, V

def load_policy(policy_csv, m, Tb, Te):
    T = Te - Tb
    actions = -np.ones((m, T), dtype=int)
    with open(policy_csv, 'r', encoding='utf-8', newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is not None and (len(header) >= 3):
            try:
                int(header[0]); int(header[1]); int(header[2])
                t,u,s = map(int, header[:3])
                if 0 <= (t-Tb) < T and 0 <= u < m:
                    actions[u, t-Tb] = s
            except ValueError:
                pass
        for row in r:
            if not row: continue
            try:
                t,u,s = map(int, row[:3])
            except ValueError:
                continue
            tr = t - Tb
            if 0 <= tr < T and 0 <= u < m:
                actions[u, tr] = s
    return actions

def replay_policy(n, m, T, cap, V, actions, user_orders, fill_missing='none'):
    """Replay a fixed policy with persistent, time-first capacity accounting."""
    if user_orders.shape != (T, m):
        raise ValueError(
            f"Bad UE-order shape: got {user_orders.shape}, expected {(T, m)}"
        )
    used = np.zeros((n, T), dtype=int)
    occupancy = np.zeros(n, dtype=int)
    handovers = np.zeros(m, dtype=int)
    blocked = np.zeros(m, dtype=int)
    last_connected = -np.ones(m, dtype=int)
    last_chosen = -np.ones(m, dtype=int)

    for t in range(T):
        for u, satellite in enumerate(last_connected):
            if satellite >= 0 and V[satellite, t] == 0:
                occupancy[satellite] -= 1
                last_connected[u] = -1
        used[:, t] = occupancy
        order = user_orders[t]
        if sorted(int(u) for u in order) != list(range(m)):
            raise ValueError(f"UE order at t={t} is not a permutation of 0..{m - 1}")
        for raw_user in order:
            u = int(raw_user)
            a = actions[u, t]
            if a < 0 and fill_missing == 'stay' and last_chosen[u] >= 0:
                a = last_chosen[u]

            legal = False
            if 0 <= a < n:
                if V[a, t] == 1 and (
                    a == last_connected[u] or occupancy[a] < cap[a]
                ):
                    legal = True

            if legal:
                previous = int(last_connected[u])
                if a != previous:
                    occupancy[a] += 1
                    if previous != -1:
                        occupancy[previous] -= 1
                if previous != -1 and a != previous:
                    handovers[u] += 1
                last_connected[u] = a
            else:
                blocked[u] += 1
                if a < 0:
                    previous = int(last_connected[u])
                    if previous != -1:
                        occupancy[previous] -= 1
                    last_connected[u] = -1

            used[:, t] = occupancy
            expected = np.bincount(
                last_connected[last_connected >= 0], minlength=n
            )
            if not np.array_equal(occupancy, expected):
                raise AssertionError("Occupancy does not match active associations")
            if np.any(occupancy < 0) or np.any(occupancy > cap):
                raise AssertionError("Occupancy is outside [0, capacity]")

            if a >= 0:
                last_chosen[u] = a

    return used, handovers, blocked, last_connected, last_chosen


def evaluate(input_path, policy_csv, fill_missing='none', order_seed=42):
    n, m, Tb, Te, T, cap, V = load_env(input_path)
    actions = load_policy(policy_csv, m, Tb, Te)
    user_orders = generate_ue_orders(m, T, order_seed)
    used, handovers, blocked, _, _ = replay_policy(
        n, m, T, cap, V, actions, user_orders, fill_missing=fill_missing
    )

    block_rates = blocked / float(T)

    # Average only over users who were connected for the full window (no blocking).
    full_connected_mask = (blocked == 0)
    included_count = int(full_connected_mask.sum())
    if included_count > 0:
        avg_handover_full = float(handovers[full_connected_mask].mean())
        avg_block_full = float(block_rates[full_connected_mask].mean())
    else:
        avg_handover_full = 0.0
        avg_block_full = 0.0

    # Keep overall averages for reference.
    avg_handover_all = float(handovers.mean())
    avg_block_all = float(block_rates.mean())

    max_u = int(handovers.argmax())
    result = {
        "max_handover": int(handovers[max_u]),
        "max_handover_user": max_u,
        "block_rate_of_max": float(block_rates[max_u]),
        "avg_handover_full": avg_handover_full,
        "avg_block_full": avg_block_full,
        "avg_handover_all": avg_handover_all,
        "avg_block_all": avg_block_all,
        "included_users_for_full": included_count,
        "T": T, "m": m, "n": n, "order_seed": order_seed
    }
    return result

def main():
    ap = argparse.ArgumentParser(description="Evaluate policy CSV into handover and blocking metrics.")
    ap.add_argument("--input", required=True, help="path to input.txt")
    ap.add_argument("--policy", required=True, help="policy CSV with columns (t,user,sat)")
    ap.add_argument("--fill_missing", choices=["none","stay"], default="none",
                    help="If no action at some t: 'stay' repeats last chosen sat; default 'none' treats as blocked if illegal")
    ap.add_argument(
        "--order_seed", type=int, default=42,
        help="seed for the fixed per-slot UE permutations; match policy export",
    )
    args = ap.parse_args()

    res = evaluate(
        args.input,
        args.policy,
        fill_missing=args.fill_missing,
        order_seed=args.order_seed,
    )
    print(f"n={res['n']} m={res['m']} T={res['T']} order_seed={res['order_seed']}")
    print(f"最大換手次數 = {res['max_handover']}（使用者 {res['max_handover_user']}），該使用者阻斷率 = {res['block_rate_of_max']:.6f}")
    print(f"平均換手次數（只計全時段連線者，共 {res['included_users_for_full']} 人） = {res['avg_handover_full']:.6f}")
    print(f"平均阻斷率（只計全時段連線者） = {res['avg_block_full']:.6f}")
    print(f"(參考) 全部使用者平均換手 = {res['avg_handover_all']:.6f}，全部平均阻斷率 = {res['avg_block_all']:.6f}")

if __name__ == "__main__":
    main()
