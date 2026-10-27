#include <bits/stdc++.h>
using namespace std;

/*
保留原始 MSH 核心風格：
1. Dijkstra 寫法不改
2. Bhandari-like path extraction / reverse edge / clear node 不改
3. fallback: i >= P.size() 時沿用 P[0]
4. evaluation 仍使用原始 MSH 風格

新增：
1. rolling window 外框
2. user start nodes
3. policy_actions.csv 輸出（t,user,sat）
4. 依原始 MSH evaluation 規則，把真正 blocking 的秒改寫成 -1
*/

#define MAX_INT INT_MAX
#define pii pair<int,int>

struct satellite {
    int begin, end, capacity;
};

struct node {
    int next, prev;
    node() { next = prev = -1; }
};

struct WindowResult {
    vector<vector<int>> policy_per_sec; // m x win_len
};

static vector<int> dijkstra(int s, int d, vector<vector<pii>> &G)
{
    vector<int> dist(G.size(), MAX_INT), p(G.size(), 0);
    vector<bool> visited(G.size(), false);
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    pq.emplace(0, 0);
    dist[s] = 0;
    while (!pq.empty())
    {
        int u = pq.top().second;
        pq.pop();
        if (visited[u]) continue;
        visited[u] = true;
        if (u == d) break;
        for (auto &[v, w] : G[u])
        {
            if (dist[u] != MAX_INT && dist[v] > dist[u] + w)
            {
                dist[v] = dist[u] + w;
                p[v] = u;
                visited[v] = false;
                pq.emplace(dist[v], v);
            }
        }
    }
    return p;
}

static vector<int> decode_sat_sequence_from_head(
    int head,
    int d,
    const vector<node>& ln,
    const vector<pii>& table
)
{
    vector<int> seq;
    int cur = head;
    while (cur != -1 && cur != d) {
        int satTag = table[cur].first;
        if (satTag > 0) {
            int sat = satTag - 1;
            if (seq.empty() || seq.back() != sat) seq.push_back(sat);
        }
        cur = ln[cur].next;
    }
    return seq;
}

static void expand_sequence_to_policy(
    const vector<satellite>& S,
    int win_b, int win_e,
    int initSat,
    const vector<int>& seq,
    vector<int>& out_policy
)
{
    int L = win_e - win_b + 1;
    out_policy.assign(L, -1);

    auto visible = [&](int sat, int t)->bool{
        return sat >= 0 && sat < (int)S.size() && S[sat].begin <= t && t <= S[sat].end;
    };

    int curSat = -1;
    if (visible(initSat, win_b)) curSat = initSat;
    else if (!seq.empty() && visible(seq[0], win_b)) curSat = seq[0];

    int t = win_b;
    int si = 0;

    if (curSat != -1) {
        for (int k = 0; k < (int)seq.size(); k++) {
            if (seq[k] == curSat) {
                si = k;
                break;
            }
        }
    }

    while (t <= win_e) {
        if (curSat == -1) {
            bool ok = false;
            for (int k = si; k < (int)seq.size(); k++) {
                if (visible(seq[k], t)) {
                    curSat = seq[k];
                    si = k;
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                for (int tt = t; tt <= win_e; tt++) out_policy[tt - win_b] = -1;
                break;
            }
        }

        int fill_until = min(S[curSat].end, win_e);
        for (int tt = t; tt <= fill_until; tt++) {
            out_policy[tt - win_b] = visible(curSat, tt) ? curSat : -1;
        }

        t = fill_until + 1;

        int nextSat = -1;
        for (int k = si + 1; k < (int)seq.size(); k++) {
            if (t <= win_e && visible(seq[k], t)) {
                nextSat = seq[k];
                si = k;
                break;
            }
        }
        curSat = nextSat;
    }
}

static WindowResult solve_msh_window_original_style(
    const vector<satellite>& S,
    int n, int m,
    int win_b, int win_e,
    const vector<int>& init_sat_of_user
)
{
    int s, d;
    int user_base = 1;

    vector<int> cnt(n);
    vector<pii> table;
    vector<vector<pii>> G;

    s = 0;
    table.emplace_back(0, 0);
    G.emplace_back(vector<pii>());

    for (int u = 0; u < m; u++) {
        table.emplace_back(0, 0);
        G.emplace_back(vector<pii>());
    }

    for (int i = 0; i < n; i++)
    {
        cnt[i] = (int)G.size();
        for (int c = 0; c < S[i].capacity; c++)
        {
            table.emplace_back(i + 1, c + 1);
            G.emplace_back(vector<pii>());
            G[(int)G.size() - 1].emplace_back((int)G.size(), 0);
            table.emplace_back(i + 1, c + 1);
            G.emplace_back(vector<pii>());
        }
    }

    cnt.emplace_back((int)G.size());
    d = (int)G.size();
    table.emplace_back(0, 0);
    G.emplace_back(vector<pii>());

    auto visible = [&](int sat, int t)->bool{
        return S[sat].begin <= t && t <= S[sat].end;
    };

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j) continue;
            if (S[j].begin <= S[i].end && S[i].end <= S[j].end)
                for (int k = cnt[i] + 1; k < cnt[i + 1]; k += 2)
                    for (int l = cnt[j]; l < cnt[j + 1]; l += 2)
                        G[k].emplace_back(l, 1);
        }

        if (S[i].begin <= win_e && win_e <= S[i].end)
            for (int k = cnt[i] + 1; k < cnt[i + 1]; k += 2)
                G[k].emplace_back(d, 0);
    }

    for (int u = 0; u < m; u++) {
        int usrc = user_base + u;
        int initSat = init_sat_of_user[u];

        if (initSat >= 0 && initSat < n && visible(initSat, win_b)) {
            for (int k = cnt[initSat]; k < cnt[initSat + 1]; k += 2)
                G[usrc].emplace_back(k, 0);
        } else {
            for (int i = 0; i < n; i++) {
                if (S[i].begin <= win_b && win_b <= S[i].end)
                    for (int k = cnt[i]; k < cnt[i + 1]; k += 2)
                        G[usrc].emplace_back(k, 0);
            }
        }

        G[s].emplace_back(usrc, 0);
    }

    int m1;
    vector<int> P;
    vector<node> linked(G.size());

    for (m1 = 0; m1 < m; m1++)
    {
        vector<int> p = dijkstra(s, d, G);
        int cur = d, prev = p[d];

        if (prev == 0 && d != s) {
            bool has_edge = false;
            for (auto &e : G[0]) if (e.first == d) has_edge = true;
            if (!has_edge && p[d] == 0) {
                bool maybe_unreachable = true;
                for (auto &e : G[0]) {
                    if (e.first != d) { maybe_unreachable = false; break; }
                }
                if (maybe_unreachable) break;
            }
        }

        while (cur != s)
        {
            bool flag = false;
            for (auto it = G[prev].begin(); it != G[prev].end(); it++)
            {
                if ((prev == s && (*it).first == cur))
                {
                    if ((*it).second < 0) flag = true;
                    G[cur].emplace_back(prev, -(*it).second);
                    G[prev].erase(it);
                    break;
                }
                else if (prev != s)
                {
                    G[cur].emplace_back(prev, -(*it).second);
                }
            }
            if (prev != s) G[prev].clear();
            if (!flag)
            {
                linked[prev].next = cur;
                linked[cur].prev = prev;
            }
            cur = prev;
            prev = p[prev];
            if (cur != prev && prev == s) P.emplace_back(cur);
        }
    }

    WindowResult wr;
    int win_len = win_e - win_b + 1;
    wr.policy_per_sec.assign(m, vector<int>(win_len, -1));

    for (int u = 0; u < m; u++) {
        int head = -1;
        if (!P.empty()) {
            if (u >= (int)P.size()) head = P[0];
            else head = P[u];
        }

        vector<int> seq;
        if (head != -1) seq = decode_sat_sequence_from_head(head, d, linked, table);

        vector<int> per_sec;
        expand_sequence_to_policy(S, win_b, win_e, init_sat_of_user[u], seq, per_sec);
        wr.policy_per_sec[u] = std::move(per_sec);
    }

    return wr;
}

// 依原本 evaluation 規則，把真正 blocking 的秒改成 -1
static void apply_blocking_mask_msh_style(
    const vector<satellite>& S,
    int n, int m,
    int T_b, int T_e,
    vector<vector<int>>& policy
){
    int T_total = T_e - T_b + 1;

    struct Seg {
        int sat;
        int start_t;
        int end_t;
    };

    vector<vector<Seg>> user_segs(m);

    for (int u = 0; u < m; u++) {
        int last = -2;
        int seg_start_idx = -1;

        for (int t = 0; t < T_total; t++) {
            int sat = policy[t][u];
            if (t == 0) {
                last = sat;
                seg_start_idx = 0;
            } else if (sat != last) {
                if (last >= 0 && last < n) {
                    user_segs[u].push_back({last, T_b + seg_start_idx, T_b + t - 1});
                }
                seg_start_idx = t;
                last = sat;
            }
        }

        if (last >= 0 && last < n) {
            user_segs[u].push_back({last, T_b + seg_start_idx, T_b + T_total - 1});
        }
    }

    vector<int> cap_rem(n);
    for (int i = 0; i < n; i++) cap_rem[i] = S[i].capacity;

    for (int u = 0; u < m; u++) {
        auto& segs = user_segs[u];
        if (segs.empty()) continue;

        int v0 = segs[0].sat;
        if (--cap_rem[v0] < 0) {
            int block_start = segs[0].start_t;
            int block_end   = min(S[v0].end, T_e);
            for (int tt = block_start; tt <= block_end; tt++) {
                policy[tt - T_b][u] = -1;
            }
        }

        for (int k = 1; k < (int)segs.size(); k++) {
            int curSat = segs[k].sat;
            if (--cap_rem[curSat] < 0) {
                int block_start = segs[k].start_t;
                int block_end   = min(S[curSat].end, T_e);
                for (int tt = block_start; tt <= block_end; tt++) {
                    policy[tt - T_b][u] = -1;
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string policy_filename = "policy_actions.csv";
    if (argc >= 2) policy_filename = argv[1];

    int n, m, T_b, T_e;
    if (!(cin >> n >> m >> T_b >> T_e)) {
        cerr << "Error: failed to read first line (n m T_b T_e).\n";
        return 1;
    }

    vector<satellite> S(n);
    for (int i = 0; i < n; i++) cin >> S[i].begin >> S[i].end >> S[i].capacity;

    const int W = 300;
    const int H = 150;

    int T_total = T_e - T_b + 1;
    vector<vector<int>> policy(T_total, vector<int>(m, -1));
    vector<int> init_sat_of_user(m, -1);

    int first_b = T_b;
    int first_e = min(T_e, first_b + W - 1);
    auto wr0 = solve_msh_window_original_style(S, n, m, first_b, first_e, init_sat_of_user);

    int first_exec_len = first_e - first_b + 1;
    for (int dt = 0; dt < first_exec_len; dt++) {
        int t = first_b + dt;
        int gi = t - T_b;
        for (int u = 0; u < m; u++) policy[gi][u] = wr0.policy_per_sec[u][dt];
    }

    for (int u = 0; u < m; u++) init_sat_of_user[u] = wr0.policy_per_sec[u][first_exec_len - 1];

    int next_write_t = first_b + first_exec_len;

    for (int cur_b = T_b + H; cur_b <= T_e && next_write_t <= T_e; cur_b += H) {
        int cur_e = min(T_e, cur_b + W - 1);
        auto wr = solve_msh_window_original_style(S, n, m, cur_b, cur_e, init_sat_of_user);

        int write_from = next_write_t;
        int write_to   = min(T_e, next_write_t + H - 1);

        for (int t = write_from; t <= write_to; t++) {
            int gi = t - T_b;
            int local_idx = t - cur_b;
            if(local_idx < 0 || local_idx >= (int)wr.policy_per_sec[0].size()) continue;
            for (int u = 0; u < m; u++) policy[gi][u] = wr.policy_per_sec[u][local_idx];
        }

        int last_t = write_to;
        int last_local = last_t - cur_b;
        if (0 <= last_local && last_local < (int)wr.policy_per_sec[0].size()) {
            for (int u = 0; u < m; u++) init_sat_of_user[u] = wr.policy_per_sec[u][last_local];
        }

        next_write_t += H;
    }

    apply_blocking_mask_msh_style(S, n, m, T_b, T_e, policy);

    vector<vector<int>> seq_user(m);
    for (int u = 0; u < m; u++) {
        int last = -2;
        for (int t = 0; t < T_total; t++) {
            int sat = policy[t][u];
            if (sat == last) continue;
            last = sat;
            if (sat >= 0 && sat < n) {
                if (seq_user[u].empty() || seq_user[u].back() != sat) seq_user[u].push_back(sat);
            }
        }
    }

    vector<long long> blocking_rate(n, 0);
    vector<int> cap_rem(n);
    for (int i = 0; i < n; i++) cap_rem[i] = S[i].capacity;

    int mx = 0;
    long long sum = 0;

    for (int i = 0; i < m; i++)
    {
        auto &seq = seq_user[i];
        if (seq.empty()) {
            mx = max(mx, 0);
            continue;
        }

        int cnt = (int)seq.size() - 1;
        mx = max(mx, cnt);
        sum += cnt;

        int v0 = seq[0];
        if (--cap_rem[v0] < 0) blocking_rate[v0] += S[v0].end - T_b;

        for (int k = 1; k < (int)seq.size(); k++)
        {
            int uSat = seq[k - 1];
            int vSat = seq[k];
            if (--cap_rem[vSat] < 0) blocking_rate[vSat] += S[vSat].end - S[uSat].end;
        }
    }

    long long mx_blocking_rate = 0, sum_blocking_rate = 0;
    for (int i = 0; i < n; i++)
    {
        mx_blocking_rate = max(mx_blocking_rate, blocking_rate[i]);
        sum_blocking_rate += blocking_rate[i];
    }

    cout << mx << ' ' << fixed << setprecision(2)
         << (double(mx_blocking_rate) / double(T_e - T_b + 1)) << '\n';

    cout << fixed << setprecision(2)
         << (double(sum) / double(m)) << ' '
         << (double(sum_blocking_rate) / double(T_e - T_b + 1)) << '\n';

    ofstream fout(policy_filename);
    if (!fout.is_open()) {
        cerr << "Error: cannot write " << policy_filename << "\n";
        return 1;
    }

    fout << "t,user,sat\n";
    for (int u = 0; u < m; u++) {
        for (int t = 0; t < T_total; t++) {
            fout << t << "," << u << "," << policy[t][u] << "\n";
        }
    }
    fout.close();

    cerr << "Saved policy file: " << policy_filename << "\n";
    return 0;
}