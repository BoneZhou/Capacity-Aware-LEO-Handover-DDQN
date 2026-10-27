#include <bits/stdc++.h>
using namespace std;

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
    vector<vector<int>> policy_per_sec;
};

static inline int node_sat_id(const vector<pair<int,int>>& table, int idx) {
    if (idx < 0 || idx >= (int)table.size()) return -1;
    int tag = table[idx].first;
    if (tag <= 0) return -1;
    return tag - 1;
}

static inline int node_sat_end(
    const vector<pair<int,int>>& table,
    const vector<satellite>& S,
    int idx
){
    int sid = node_sat_id(table, idx);
    if (sid < 0 || sid >= (int)S.size()) return -1;
    return S[sid].end;
}

static vector<int> dijkstra_parent(
    int s, int d,
    vector<vector<pii>> &G,
    const vector<pair<int,int>>& table,
    const vector<satellite>& S_full
) {
    vector<int> dist((int)G.size(), MAX_INT), p((int)G.size(), -1);
    vector<int> best_end((int)G.size(), -1);
    vector<bool> visited((int)G.size(), false);

    struct State {
        int dist;
        int end_pref;
        int node;
    };
    struct Cmp {
        bool operator()(const State& a, const State& b) const {
            if (a.dist != b.dist) return a.dist > b.dist;
            if (a.end_pref != b.end_pref) return a.end_pref < b.end_pref;
            return a.node > b.node;
        }
    };

    priority_queue<State, vector<State>, Cmp> pq;
    dist[s] = 0;
    best_end[s] = INT_MAX;
    pq.push({0, INT_MAX, s});

    while (!pq.empty()) {
        State cur = pq.top(); pq.pop();
        int u = cur.node;
        if (visited[u]) continue;
        visited[u] = true;
        if (u == d) break;

        for (auto &e : G[u]) {
            int v = e.first, w = e.second;
            if (dist[u] == MAX_INT) continue;

            int cand_dist = dist[u] + w;
            int cand_end = node_sat_end(table, S_full, v);

            if (cand_dist < dist[v]) {
                dist[v] = cand_dist;
                p[v] = u;
                best_end[v] = cand_end;
                visited[v] = false;
                pq.push({dist[v], best_end[v], v});
            }
            else if (cand_dist == dist[v] && cand_end > best_end[v]) {
                p[v] = u;
                best_end[v] = cand_end;
                visited[v] = false;
                pq.push({dist[v], best_end[v], v});
            }
        }
    }
    return p;
}

static vector<int> decode_sat_sequence_from_head(
    int head,
    int d,
    const vector<node>& ln,
    const vector<pair<int,int>>& table
){
    vector<int> seq;
    int cur = head;
    while (cur != -1 && cur != d) {
        int satTag = table[cur].first;
        if (satTag > 0) {
            int sat_id = satTag - 1;
            if (seq.empty() || seq.back() != sat_id) seq.push_back(sat_id);
        }
        cur = ln[cur].next;
    }
    return seq;
}

static void expand_sequence_to_policy(
    const vector<satellite>& S_full,
    int win_b, int win_e,
    int initSat,
    const vector<int>& seq,
    vector<int>& out_policy_per_sec
){
    int L = win_e - win_b + 1;
    out_policy_per_sec.assign(L, -1);

    auto visible = [&](int sat, int t)->bool{
        return sat>=0 && sat<(int)S_full.size() && S_full[sat].begin <= t && t <= S_full[sat].end;
    };

    int curSat = -1;
    if (initSat >= 0 && initSat < (int)S_full.size() && visible(initSat, win_b)) curSat = initSat;
    else if (!seq.empty() && visible(seq[0], win_b)) curSat = seq[0];

    int t = win_b;
    int si = 0;

    if (curSat != -1) {
        int found = -1;
        for (int k=0; k<(int)seq.size(); k++){
            if (seq[k] == curSat) { found = k; break; }
        }
        if (found != -1) si = found;
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
                for (int tt=t; tt<=win_e; tt++) out_policy_per_sec[tt-win_b] = -1;
                break;
            }
        }

        int fill_until = min(S_full[curSat].end, win_e);
        for (int tt=t; tt<=fill_until; tt++) {
            out_policy_per_sec[tt-win_b] = visible(curSat, tt) ? curSat : -1;
        }

        t = fill_until + 1;

        int nextSat = -1;
        for (int k = si+1; k < (int)seq.size(); k++) {
            if (t <= win_e && visible(seq[k], t)) {
                nextSat = seq[k];
                si = k;
                break;
            }
        }
        curSat = nextSat;
    }
}

static WindowResult solve_msh_window(
    const vector<satellite>& S_full,
    int n, int local_m,
    int win_b, int win_e,
    const vector<int>& init_sat_local,
    const vector<int>& reserved
){
    int s = 0;
    int user_base = 1;

    vector<int> cnt(n+1, 0);
    vector<pair<int,int>> table;
    vector<vector<pii>> G;

    table.emplace_back(0,0);
    G.emplace_back();

    for (int u=0; u<local_m; u++){
        table.emplace_back(0,0);
        G.emplace_back();
    }

    for (int i=0; i<n; i++){
        cnt[i] = (int)G.size();

        int eff_cap = S_full[i].capacity - reserved[i];
        if (eff_cap < 0) eff_cap = 0;

        for (int c=0; c<eff_cap; c++){
            table.emplace_back(i+1, c+1);
            G.emplace_back();
            table.emplace_back(i+1, c+1);
            G.emplace_back();

            int in_idx = (int)G.size() - 2;
            int out_idx = (int)G.size() - 1;
            G[in_idx].emplace_back(out_idx, 0);
        }
        cnt[i+1] = (int)G.size();
    }

    int d = (int)G.size();
    table.emplace_back(0,0);
    G.emplace_back();

    auto sat_in_range = [&](int sat, int t)->bool{
        return S_full[sat].begin <= t && t <= S_full[sat].end;
    };

    vector<int> sat_order(n);
    iota(sat_order.begin(), sat_order.end(), 0);
    stable_sort(sat_order.begin(), sat_order.end(), [&](int a, int b){
        if (S_full[a].end != S_full[b].end) return S_full[a].end > S_full[b].end;
        if (S_full[a].begin != S_full[b].begin) return S_full[a].begin < S_full[b].begin;
        return a < b;
    });

    for (int i=0; i<n; i++){
        if (cnt[i] == cnt[i+1]) continue;

        for (int ord=0; ord<n; ord++){
            int j = sat_order[ord];
            if (i == j) continue;
            if (cnt[j] == cnt[j+1]) continue;

            if (S_full[j].begin <= S_full[i].end && S_full[i].end <= S_full[j].end){
                for (int k = cnt[i] + 1; k < cnt[i+1]; k += 2){
                    for (int l = cnt[j]; l < cnt[j+1]; l += 2){
                        G[k].emplace_back(l, 1);
                    }
                }
            }
        }

        if (sat_in_range(i, win_e)){
            for (int k = cnt[i] + 1; k < cnt[i+1]; k += 2){
                G[k].emplace_back(d, 0);
            }
        }
    }

    for (int u=0; u<local_m; u++){
        int usrc = user_base + u;
        int initSat = init_sat_local[u];

        auto connect_all_visible = [&](){
            for (int ord=0; ord<n; ord++){
                int i = sat_order[ord];
                if (cnt[i] == cnt[i+1]) continue;
                if (sat_in_range(i, win_b)){
                    for (int k = cnt[i]; k < cnt[i+1]; k += 2){
                        G[usrc].emplace_back(k, 0);
                    }
                }
            }
        };

        if (initSat >= 0 && initSat < n && sat_in_range(initSat, win_b) && cnt[initSat] != cnt[initSat+1]){
            for (int k = cnt[initSat]; k < cnt[initSat+1]; k += 2){
                G[usrc].emplace_back(k, 0);
            }
        } else {
            connect_all_visible();
        }

        G[s].emplace_back(usrc, 0);
    }

    vector<node> ln(G.size());
    vector<int> heads;

    for (int flow=0; flow<local_m; flow++){
        vector<int> p = dijkstra_parent(s, d, G, table, S_full);
        if (p[d] == -1) break;

        int cur = d;
        int prev = p[d];

        while (cur != s && prev != -1) {
            bool flag = false;

            for (auto it = G[prev].begin(); it != G[prev].end(); ++it) {
                if (it->first == cur) {
                    if (it->second < 0) flag = true;
                    G[cur].emplace_back(prev, -it->second);
                    G[prev].erase(it);
                    break;
                }
            }

            if (prev != s) G[prev].clear();

            if (!flag) {
                ln[prev].next = cur;
                ln[cur].prev = prev;
            }

            cur = prev;
            prev = p[prev];

            if (prev == s && cur != s) {
                int h = cur;
                if (h >= user_base && h < user_base + local_m) {
                    if (ln[h].next != -1) heads.push_back(ln[h].next);
                } else {
                    heads.push_back(h);
                }
            }
        }
    }

    WindowResult wr;
    int len = win_e - win_b + 1;
    wr.policy_per_sec.assign(local_m, vector<int>(len, -1));

    for (int u=0; u<local_m; u++){
        int head = -1;
        if (!heads.empty()) head = (u < (int)heads.size()) ? heads[u] : heads[0];

        vector<int> seq;
        if (head != -1) seq = decode_sat_sequence_from_head(head, d, ln, table);

        vector<int> per_sec;
        expand_sequence_to_policy(S_full, win_b, win_e, init_sat_local[u], seq, per_sec);
        wr.policy_per_sec[u] = std::move(per_sec);
    }

    return wr;
}

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
            for (int tt = block_start; tt <= block_end; tt++) policy[tt - T_b][u] = -1;
        }

        for (int k = 1; k < (int)segs.size(); k++) {
            int curSat = segs[k].sat;
            if (--cap_rem[curSat] < 0) {
                int block_start = segs[k].start_t;
                int block_end   = min(S[curSat].end, T_e);
                for (int tt = block_start; tt <= block_end; tt++) policy[tt - T_b][u] = -1;
            }
        }
    }
}

int main(int argc, char* argv[]){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string policy_filename = "policy_actions.csv";
    if (argc >= 2) policy_filename = argv[1];

    int n, m, T_b, T_e;
    if(!(cin >> n >> m >> T_b >> T_e)){
        cerr << "Error: failed to read first line (n m T_b T_e).\n";
        return 1;
    }

    vector<satellite> S(n);
    for(int i=0;i<n;i++) cin >> S[i].begin >> S[i].end >> S[i].capacity;

    const int W = 3600;
    const int H = 300;

    int T_total = T_e - T_b + 1;
    vector<vector<int>> policy(T_total, vector<int>(m, -1));
    vector<int> curSat(m, -1);

    auto visible = [&](int sat, int t)->bool{
        return sat >= 0 && sat < n && S[sat].begin <= t && t <= S[sat].end;
    };

    int cur_b = T_b;
    while(cur_b <= T_e){
        int cur_e = min(T_e, cur_b + W - 1);
        int exec_len = min(H, cur_e - cur_b + 1);
        int write_to = cur_b + exec_len - 1;

        for(int t=cur_b; t<=write_to; t++){
            int gi = t - T_b;
            for(int u=0; u<m; u++) policy[gi][u] = -1;
        }

        vector<int> active_users;
        vector<int> init_sat_local;
        vector<int> reserved(n, 0);

        for(int u=0; u<m; u++){
            if(curSat[u] >= 0 && curSat[u] < n && visible(curSat[u], cur_b)){
                int t_hold_end = min(write_to, S[curSat[u]].end);

                for(int t=cur_b; t<=t_hold_end; t++) policy[t - T_b][u] = curSat[u];

                if(t_hold_end == write_to){
                    reserved[curSat[u]]++;
                } else {
                    active_users.push_back(u);
                    init_sat_local.push_back(-1);
                }
            } else {
                active_users.push_back(u);
                init_sat_local.push_back(curSat[u]);
            }
        }

        WindowResult wr;
        if(!active_users.empty()){
            wr = solve_msh_window(
                S, n, (int)active_users.size(),
                cur_b, cur_e,
                init_sat_local,
                reserved
            );
        }

        for(int k=0; k<(int)active_users.size(); k++){
            int u = active_users[k];
            for(int t=cur_b; t<=write_to; t++){
                int gi = t - T_b;
                if(policy[gi][u] != -1) continue;
                int local = t - cur_b;
                if(local >= 0 && local < (int)wr.policy_per_sec[k].size()){
                    policy[gi][u] = wr.policy_per_sec[k][local];
                }
            }
        }

        for(int u=0; u<m; u++) curSat[u] = policy[write_to - T_b][u];
        cur_b += H;
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

    int mx_handover = 0;
    long long sum_handover = 0;

    for (int u = 0; u < m; u++) {
        auto &seq = seq_user[u];

        if (seq.empty()) {
            mx_handover = max(mx_handover, 0);
            continue;
        }

        int handover_cnt = (int)seq.size() - 1;
        mx_handover = max(mx_handover, handover_cnt);
        sum_handover += handover_cnt;

        int v0 = seq[0];
        if (--cap_rem[v0] < 0) {
            long long add = (long long)S[v0].end - (long long)T_b;
            if (add > 0) blocking_rate[v0] += add;
        }

        for (int k = 1; k < (int)seq.size(); k++) {
            int uSat = seq[k-1];
            int vSat = seq[k];
            if (--cap_rem[vSat] < 0) {
                long long add = (long long)S[vSat].end - (long long)S[uSat].end;
                if (add > 0) blocking_rate[vSat] += add;
            }
        }
    }

    long long mx_blocking = 0;
    long long sum_blocking = 0;
    for (int i = 0; i < n; i++) {
        mx_blocking = max(mx_blocking, blocking_rate[i]);
        sum_blocking += blocking_rate[i];
    }

    double denom = double(T_e - T_b + 1);

    cout << mx_handover << ' ' << fixed << setprecision(2)
         << (denom > 0 ? double(mx_blocking) / denom : 0.0) << '\n';

    cout << fixed << setprecision(2)
         << (m > 0 ? double(sum_handover) / double(m) : 0.0) << ' '
         << (denom > 0 ? double(sum_blocking) / denom : 0.0) << '\n';

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