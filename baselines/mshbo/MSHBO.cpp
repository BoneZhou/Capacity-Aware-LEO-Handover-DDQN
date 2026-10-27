#include <bits/stdc++.h>
using namespace std;

// Rolling-window (receding-horizon) wrapper around the original MSHBO core.
// Input format:
//   line1: n  m  T_b  T_e
//   next n lines: begin  end  capacity   (satellite id = 0..n-1)
//
// Output:
//   policy.csv : t,user,sat
//                先輸出 user 0 的所有秒數，再 user 1 的所有秒數...
//                例如：
//                t,user,sat
//                0,0,5
//                1,0,5
//                ...
//                3600,0,7
//                0,1,3
//                1,1,-1
//                ...
//
//   stdout:
//     max_handover  max_blocking_rate_per_user
//     avg_handover_per_user  avg_blocking_over_all_users

#define INF_INT INT_MAX
using pii = pair<int,int>;

struct Satellite {
    int begin = 0, end = 0, capacity = 0;
};

struct NodeNext {
    int next = -1;
};

struct SolveResult {
    // sat_id per user per local time index (0..len-1); -1 means blocked
    vector<vector<int>> sat;
    // for blocked slots, which satellite the blocking node belongs to (0..n-1), else -1
    vector<vector<int>> block_on_sat;
};

static vector<int> dijkstra_parent(int s, int d, const vector<vector<pii>>& G) {
    vector<int> dist(G.size(), INF_INT), parent(G.size(), -1);
    vector<char> vis(G.size(), 0);
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[s] = 0;
    pq.emplace(0, s);

    while (!pq.empty()) {
        auto [du, u] = pq.top();
        pq.pop();

        if (vis[u]) continue;
        vis[u] = 1;
        if (u == d) break;
        if (du != dist[u]) continue;

        for (const auto& [v, w] : G[u]) {
            if (dist[u] != INF_INT && dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }
    return parent;
}

static SolveResult solve_window(
    const vector<Satellite>& S,
    int n, int m,
    int win_b, int win_e,
    const vector<int>& init_sat_of_user
){
    // label sweep to estimate actual_m in [win_b, win_e]
    vector<pair<int,int>> label;
    int C_max = 0;
    label.reserve(2 * n);

    for (int i = 0; i < n; i++) {
        C_max = max(C_max, S[i].capacity);
        label.emplace_back(S[i].begin, -S[i].capacity);
        label.emplace_back(S[i].end,   +S[i].capacity);
    }
    sort(label.begin(), label.end());

    int cnt = 0;
    int actual_m = INT_MAX;
    for (int i = 0; i < (int)label.size(); i++) {
        cnt -= label[i].second;
        if (label[i].first <= win_b) actual_m = cnt;
        else if (label[i].first < win_e) actual_m = min(actual_m, cnt);
    }
    if (actual_m == INT_MAX) actual_m = 0;

    // mm follows original logic
    int mm = max(0, m - actual_m + 1);

    // Node indexing plan:
    // 0 = super source
    // 1..m = user sources
    // then satellite time-expanded nodes
    int s = 0;
    int base = 1 + m; // first satellite node index

    // Build SG/Sn and total node count
    vector<int> SG(n + 1);
    vector<int> Sn(n);
    int G_n = base;

    for (int i = 0; i < n; i++) {
        SG[i] = G_n;
        Sn[i] = 2 * (S[i].capacity + mm); // same as original
        long long add = 1LL * (S[i].end - S[i].begin + 1) * Sn[i];
        G_n += (int)add;
    }
    SG[n] = G_n;

    int d = G_n; // sink
    vector<vector<pii>> G(d + 1);

    // meta
    vector<int> meta_sat(d + 1, -1);
    vector<int> meta_time(d + 1, INT_MIN);
    vector<int> meta_type(d + 1, -1);
    vector<char> blocking_flag(d + 1, 0);

    auto set_meta = [&](int idx, int sat, int time, int type) {
        if (idx < 0 || idx > d) return;
        if (meta_sat[idx] == -1) meta_sat[idx] = sat;
        if (meta_time[idx] == INT_MIN) meta_time[idx] = time;
        if (meta_type[idx] == -1) meta_type[idx] = type;
    };

    auto node_index = [&](int sat, int time, int type, int id) -> int {
        // type: 0 in(nonblock), 1 out(nonblock), 2 in(block), 3 out(block)
        int cap = S[sat].capacity;
        int offset = min(type, 2) * cap + ((type == 3) ? mm : 0) + id;
        return SG[sat] + Sn[sat] * (time - S[sat].begin) + offset;
    };

    auto add_edge = [&](int sat1,int time1,int type1,int id1,
                        int sat2,int time2,int type2,int id2,
                        int w) {
        int u = node_index(sat1, time1, type1, id1);
        int v = node_index(sat2, time2, type2, id2);
        if (u < 0 || u > d || v < 0 || v > d) return;
        G[u].emplace_back(v, w);
        set_meta(u, sat1, time1, type1);
        set_meta(v, sat2, time2, type2);
    };

    // User-source edges: s -> user_u
    for (int u = 0; u < m; u++) {
        G[s].emplace_back(1 + u, 0);
    }

    // Build edges
    for (int i = 0; i < n; i++) {
        // Inter-satellite transitions
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            for (int t = max(S[i].begin, win_b); t <= min(S[i].end, win_e - 1); t++) {
                if (S[j].begin <= t && t <= S[j].end &&
                    S[j].begin <= t + 1 && t + 1 <= S[j].end) {

                    for (int k = 0; k < S[i].capacity; k++) {
                        for (int l = 0; l < S[j].capacity; l++) {
                            add_edge(i, t, 1, k, j, t + 1, 0, l, 1);
                        }
                        for (int l = 0; l < mm; l++) {
                            add_edge(i, t, 1, k, j, t + 1, 2, l, 1);
                            int idx = node_index(j, t + 1, 2, l);
                            if (idx >= 0 && idx <= d) blocking_flag[idx] = 1;
                        }
                    }

                    for (int k = 0; k < mm; k++) {
                        for (int l = 0; l < S[j].capacity; l++) {
                            add_edge(i, t, 3, k, j, t + 1, 0, l, 1);
                        }
                        for (int l = 0; l < mm; l++) {
                            add_edge(i, t, 3, k, j, t + 1, 2, l, 1);
                            int idx = node_index(j, t + 1, 2, l);
                            if (idx >= 0 && idx <= d) blocking_flag[idx] = 1;
                        }
                    }
                }
            }
        }

        // in -> out (non-block)
        for (int t = max(S[i].begin, win_b); t <= min(S[i].end, win_e); t++) {
            for (int j = 0; j < S[i].capacity; j++) {
                add_edge(i, t, 0, j, i, t, 1, j, 0);
            }
        }

        // block in -> block out
        for (int t = max(S[i].begin, win_b); t <= min(S[i].end, win_e); t++) {
            for (int j = 0; j < mm; j++) {
                add_edge(i, t, 2, j, i, t, 3, j, (m * (win_e - win_b + 1) + 1));
            }
        }

        // stay on same sat (out -> next in)
        for (int t = max(S[i].begin, win_b); t < min(S[i].end, win_e); t++) {
            for (int j = 0; j < S[i].capacity; j++) {
                add_edge(i, t, 1, j, i, t + 1, 0, j, 0);
            }
            for (int j = 0; j < mm; j++) {
                add_edge(i, t, 3, j, i, t + 1, 2, j, 0);
                int idx = node_index(i, t + 1, 2, j);
                if (idx >= 0 && idx <= d) blocking_flag[idx] = 1;
            }
        }

        // sink connections at win_e
        if (S[i].begin <= win_e && win_e <= S[i].end) {
            for (int j = 0; j < S[i].capacity; j++) {
                int out_idx = node_index(i, win_e, 1, j);
                if (out_idx >= 0 && out_idx <= d) G[out_idx].emplace_back(d, 0);
                set_meta(out_idx, i, win_e, 1);
            }
            for (int j = 0; j < mm; j++) {
                int outb_idx = node_index(i, win_e, 3, j);
                if (outb_idx >= 0 && outb_idx <= d) G[outb_idx].emplace_back(d, 0);
                set_meta(outb_idx, i, win_e, 3);
            }
        }
    }

    // Start edges at win_b, per user constraint
    for (int u = 0; u < m; u++) {
        int usrc = 1 + u;
        int initSat = init_sat_of_user[u];

        auto connect_all_sats_at_win_b = [&]() {
            for (int i = 0; i < n; i++) {
                if (S[i].begin <= win_b && win_b <= S[i].end) {
                    // to non-block in nodes
                    for (int j = 0; j < S[i].capacity; j++) {
                        int in_idx = node_index(i, win_b, 0, j);
                        if (in_idx >= 0 && in_idx <= d) G[usrc].emplace_back(in_idx, 0);
                        set_meta(in_idx, i, win_b, 0);
                    }
                    // to block in nodes
                    for (int j = 0; j < mm; j++) {
                        int bin_idx = node_index(i, win_b, 2, j);
                        if (bin_idx >= 0 && bin_idx <= d) G[usrc].emplace_back(bin_idx, 0);
                        set_meta(bin_idx, i, win_b, 2);
                        blocking_flag[bin_idx] = 1;
                    }
                }
            }
        };

        if (initSat >= 0 && initSat < n &&
            S[initSat].begin <= win_b && win_b <= S[initSat].end) {
            // lock the start sat to initSat
            for (int j = 0; j < S[initSat].capacity; j++) {
                int in_idx = node_index(initSat, win_b, 0, j);
                if (in_idx >= 0 && in_idx <= d) G[usrc].emplace_back(in_idx, 0);
                set_meta(in_idx, initSat, win_b, 0);
            }
            for (int j = 0; j < mm; j++) {
                int bin_idx = node_index(initSat, win_b, 2, j);
                if (bin_idx >= 0 && bin_idx <= d) G[usrc].emplace_back(bin_idx, 0);
                set_meta(bin_idx, initSat, win_b, 2);
                blocking_flag[bin_idx] = 1;
            }
        } else {
            // if previously blocked or sat is no longer visible, allow re-association
            connect_all_sats_at_win_b();
        }
    }

    // Successive shortest path for m users
    vector<NodeNext> nxt(G.size());

    for (int flow = 0; flow < m; flow++) {
        vector<int> parent = dijkstra_parent(s, d, G);
        int prev = parent[d];
        if (prev == -1) {
            break;
        }

        int cur = d;
        while (cur != s) {
            int p = parent[cur];
            if (p == -1) break;

            bool reversedEdge = false;
            for (auto it = G[p].begin(); it != G[p].end(); ++it) {
                if (it->first == cur) {
                    if (it->second < 0) reversedEdge = true;
                    G[cur].emplace_back(p, -it->second);
                    G[p].erase(it);
                    break;
                }
            }
            if (!reversedEdge) nxt[p].next = cur;
            cur = p;
        }
    }

    int len = win_e - win_b + 1;
    SolveResult res;
    res.sat.assign(m, vector<int>(len, -1));
    res.block_on_sat.assign(m, vector<int>(len, -1));

    // Decode each user's routed path
    for (int u = 0; u < m; u++) {
        int usrc = 1 + u;
        if (nxt[usrc].next == -1) continue;

        int cur = usrc;
        while (cur != d && cur != -1) {
            int next = nxt[cur].next;

            if (cur >= base && cur < (int)G.size()) {
                int t = meta_time[cur];
                int sat = meta_sat[cur];
                int type = meta_type[cur];

                if (t >= win_b && t <= win_e && (type == 0 || type == 2)) {
                    int idx = t - win_b;
                    if (blocking_flag[cur]) {
                        res.sat[u][idx] = -1;
                        res.block_on_sat[u][idx] = sat;
                    } else {
                        res.sat[u][idx] = sat;
                    }
                }
            }
            cur = next;
        }
    }

    return res;
}

// 輸出成 DQN 要的格式：t,user,sat
// 排序：先 user 0 的所有時間，再 user 1 的所有時間...
static void write_policy_csv(const string& path,
                             const vector<vector<int>>& policy /*T x m*/,
                             int T_b) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "Error: cannot write " << path << "\n";
        return;
    }

    ofs << "t,user,sat\n";

    int T = (int)policy.size();
    int m = T ? (int)policy[0].size() : 0;

    for (int u = 0; u < m; u++) {
        for (int t = 0; t < T; t++) {
            int abs_t = T_b + t;
            ofs << abs_t << "," << u << "," << policy[t][u] << "\n";
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, T_b, T_e;
    if (!(cin >> n >> m >> T_b >> T_e)) {
        cerr << "Error: failed to read first line (n m T_b T_e).\n";
        return 1;
    }

    vector<Satellite> S(n);
    for (int i = 0; i < n; i++) {
        cin >> S[i].begin >> S[i].end >> S[i].capacity;
    }

    // rolling 參數
    const int W = 300;  // lookahead
    const int H = 150;  // slide

    int T_total = T_e - T_b + 1;
    vector<vector<int>> policy(T_total, vector<int>(m, -1));

    // rolling state: sat at the "last committed time" per user
    vector<int> init_sat_of_user(m, -1);

    // ---- First window: solve and commit all ----
    int first_b = T_b;
    int first_e = min(T_e, first_b + W - 1);
    auto res0 = solve_window(S, n, m, first_b, first_e, init_sat_of_user);

    int first_exec_len = first_e - first_b + 1;
    for (int dt = 0; dt < first_exec_len; dt++) {
        int t = first_b + dt;
        int gi = t - T_b;
        for (int u = 0; u < m; u++) {
            policy[gi][u] = res0.sat[u][dt];
        }
    }

    // update init to the last committed time
    for (int u = 0; u < m; u++) {
        init_sat_of_user[u] = res0.sat[u][first_exec_len - 1];
    }

    // next time index we need to fill
    int next_write_t = first_b + first_exec_len;

    // ---- Subsequent sliding windows: each time only commit new tail H seconds ----
    for (int cur_b = T_b + H; cur_b <= T_e && next_write_t <= T_e; cur_b += H) {
        int cur_e = min(T_e, cur_b + W - 1);
        auto res = solve_window(S, n, m, cur_b, cur_e, init_sat_of_user);

        int write_from = next_write_t;
        int write_to   = min(T_e, next_write_t + H - 1);

        for (int t = write_from; t <= write_to; t++) {
            int gi = t - T_b;
            int local_idx = t - cur_b;
            if (local_idx < 0 || local_idx >= (int)res.sat[0].size()) continue;

            for (int u = 0; u < m; u++) {
                policy[gi][u] = res.sat[u][local_idx];
            }
        }

        // update init to last committed second of this update
        int last_t = write_to;
        int last_local = last_t - cur_b;
        if (0 <= last_local && last_local < (int)res.sat[0].size()) {
            for (int u = 0; u < m; u++) {
                init_sat_of_user[u] = res.sat[u][last_local];
            }
        }

        next_write_t += H;
    }

    // Save policy.csv in DQN format
    write_policy_csv("policy.csv", policy, T_b);

    // -------- metrics --------
    int mx_handover = 0;
    long long sum_handover = 0;

    long long total_blocked = 0;
    int mx_blocked_user = 0;

    for (int u = 0; u < m; u++) {
        int hcnt = 0;
        int bcnt = 0;
        for (int t = 0; t < T_total; t++) {
            if (policy[t][u] == -1) bcnt++;
            if (t > 0 && policy[t][u] != policy[t - 1][u]) hcnt++;
        }
        mx_handover = max(mx_handover, hcnt);
        sum_handover += hcnt;
        total_blocked += bcnt;
        mx_blocked_user = max(mx_blocked_user, bcnt);
    }

    cout << mx_handover << ' '
         << fixed << setprecision(2)
         << (double)mx_blocked_user / (double)T_total << '\n';

    cout << fixed << setprecision(2)
         << (double)sum_handover / (double)m << ' '
         << (double)total_blocked / (double)T_total << '\n';

    return 0;
}