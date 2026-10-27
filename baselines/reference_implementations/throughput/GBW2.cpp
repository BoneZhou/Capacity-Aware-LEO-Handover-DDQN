#include <bits/stdc++.h>
using namespace std;

#define MAXD 1e100
using pdi = pair<double,int>;

struct satellite{
    int begin, end, capacity;
};

static inline bool visible_closed(const satellite& s, int t){
    return s.begin <= t && t <= s.end;
}

static vector<vector<int>> build_graph(const vector<satellite>& S, int n, int Tb, int Te){
    int s = n, d = n + 1;
    vector<vector<int>> G(n + 2);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i == j) continue;
            if(S[j].begin <= S[i].end && S[i].end < S[j].end){
                G[i].push_back(j);
            }
        }
        if(visible_closed(S[i], Tb)) G[s].push_back(i);
        if(visible_closed(S[i], Te)) G[i].push_back(d);
    }
    return G;
}

static vector<int> dijkstra_prev(
    const vector<vector<int>>& G,
    const vector<satellite>& S,
    const vector<int>& load,
    int n,
    int mx_t
){
    int s = n, d = n + 1;
    int V = n + 2;

    vector<int> prev(V, -1);
    vector<double> dist(V, MAXD);
    vector<char> vis(V, 0);

    priority_queue<pdi, vector<pdi>, greater<pdi>> pq;
    dist[s] = 0.0;
    pq.emplace(0.0, s);

    while(!pq.empty()){
        auto [du, u] = pq.top();
        pq.pop();

        if(vis[u]) continue;
        vis[u] = 1;
        if(u == d) break;

        for(int v : G[u]){
            if(v < 0 || v >= V) continue;
            if(vis[v]) continue;

            double w = 0.0;

            if(u != s && v != d){
                double term_time = (mx_t > 0)
                    ? (double)(S[u].end - S[v].begin) / (double)mx_t
                    : 0.0;

                double term_load = 0.0;
                if(load[v] > 0) term_load = 1.0 / (double)load[v];
                else            term_load = 1e5 * (double)((-load[v]) + 2);

                w = 0.7 * term_time + 0.3 * term_load;
            }
            else if(u == s && v != d){
                double term_load = 0.0;
                if(load[v] > 0) term_load = 1.0 / (double)load[v];
                else            term_load = 1e5 * (double)((-load[v]) + 2);

                w = 0.3 * term_load;
            }
            else{
                w = 0.0;
            }

            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                prev[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }

    return prev;
}

static vector<int> extract_sat_path(const vector<int>& prev, int n){
    int s = n, d = n + 1;
    vector<int> rev;

    int cur = d;
    while(cur != -1 && cur != s){
        int p = prev[cur];
        if(p == -1) break;
        if(0 <= p && p < n) rev.push_back(p);
        cur = p;
    }

    reverse(rev.begin(), rev.end());
    return rev;
}

static long long clipped_block_seconds(
    const vector<satellite>& S,
    int curSat,
    int prevEnd,
    int seg_b,
    int seg_e
){
    int raw_start = max(prevEnd, seg_b);
    int raw_end   = min(S[curSat].end, seg_e);

    if(raw_end <= raw_start) return 0;
    return (long long)(raw_end - raw_start);
}

static void apply_blocking_mask_gbw2_style(
    const vector<satellite>& S,
    int n, int m,
    int T_b, int T_e,
    vector<vector<int>>& policy
){
    int T_total = T_e - T_b + 1;
    vector<int> load(n);
    for(int i = 0; i < n; i++) load[i] = S[i].capacity;

    for(int u = 0; u < m; u++){
        int t = 0;
        while(t < T_total){
            int sat = policy[t][u];
            if(sat < 0 || sat >= n){
                t++;
                continue;
            }

            int start = t;
            while(t + 1 < T_total && policy[t + 1][u] == sat) t++;
            int end = t;

            load[sat] -= 1;
            if(load[sat] < 0){
                for(int k = start; k <= end; k++) policy[k][u] = -1;
            }
            t++;
        }
    }
}

int main(int argc, char* argv[]){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string policy_filename = "policy_actions.csv";
    if(argc >= 2) policy_filename = argv[1];

    int n, m, T_b, T_e;
    cin >> n >> m >> T_b >> T_e;

    vector<satellite> S(n);
    int mx_t = 0;

    for(int i = 0; i < n; i++){
        cin >> S[i].begin >> S[i].end >> S[i].capacity;
        mx_t = max(mx_t, S[i].end - S[i].begin);
    }

    const int W = 300;
    const int H = 150;

    vector<vector<int>> policy(T_e - T_b + 1, vector<int>(m, -1));

    vector<long long> blocking_rate(n, 0);
    vector<int> ho_cnt(m, 0);
    int mx_handover = 0;

    int decided_end = T_b - 1;
    bool first = true;

    while(decided_end < T_e){
        int seg_b = decided_end + 1;
        int k = first ? 0 : ((seg_b - (T_b + W)) / H + 1);
        int seg_e = min(T_e, T_b + W + k * H);

        auto G = build_graph(S, n, seg_b, seg_e);

        vector<int> load(n);
        for(int i = 0; i < n; i++) load[i] = S[i].capacity;

        for(int u = 0; u < m; u++){
            auto prev = dijkstra_prev(G, S, load, n, mx_t);
            auto path = extract_sat_path(prev, n);

            if(path.empty()) continue;

            int cost = max(0, (int)path.size() - 1);
            ho_cnt[u] += cost;
            mx_handover = max(mx_handover, ho_cnt[u]);

            for(int t = seg_b; t <= seg_e; t++) policy[t - T_b][u] = -1;

            int cur_start = seg_b;

            for(int idx = 0; idx < (int)path.size(); idx++){
                int sat = path[idx];

                int serve_start = cur_start;
                int serve_end   = min(seg_e, S[sat].end);

                if(serve_start <= serve_end){
                    for(int t = serve_start; t <= serve_end; t++) policy[t - T_b][u] = sat;
                }

                load[sat] -= 1;
                if(load[sat] < 0){
                    blocking_rate[sat] += clipped_block_seconds(S, sat, cur_start, seg_b, seg_e);
                }

                cur_start = S[sat].end + 1;
                if(cur_start > seg_e) break;
            }
        }

        decided_end = seg_e;
        first = false;
    }

    apply_blocking_mask_gbw2_style(S, n, m, T_b, T_e, policy);

    long long mx_blocking = 0, sum_blocking = 0;
    for(int i = 0; i < n; i++){
        mx_blocking = max(mx_blocking, blocking_rate[i]);
        sum_blocking += blocking_rate[i];
    }

    long long sum_ho = 0;
    for(int u = 0; u < m; u++) sum_ho += ho_cnt[u];

    double denom = double(T_e - T_b + 1);

    cout << mx_handover << ' ' << fixed << setprecision(2)
         << (denom > 0 ? double(mx_blocking) / denom : 0.0) << "\n";

    cout << fixed << setprecision(2)
         << (m > 0 ? double(sum_ho) / double(m) : 0.0) << ' '
         << (denom > 0 ? double(sum_blocking) / denom : 0.0) << "\n";

    ofstream fout(policy_filename);
    if(!fout){
        cerr << "Failed to open output file: " << policy_filename << "\n";
        return 1;
    }

    fout << "t,user,sat\n";
    for(int u = 0; u < m; u++){
        for(int t = T_b; t <= T_e; t++){
            fout << (t - T_b) << "," << u << "," << policy[t - T_b][u] << "\n";
        }
    }
    fout.close();

    cerr << "Saved policy file: " << policy_filename << "\n";
    return 0;
}