#include <bits/stdc++.h>
using namespace std;

#define INF 1e100
using pdi = pair<double, int>;

struct Satellite {
    int begin, end, capacity;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Input
    int n, m, T_b, T_e;
    cin >> n >> m >> T_b >> T_e;

    vector<Satellite> S(n);
    vector<int> load(n);
    int mx_t = 0;
    int mx_c = 0;

    for (int i = 0; i < n; i++) {
        cin >> S[i].begin >> S[i].end >> S[i].capacity;
        load[i] = S[i].capacity;   // available channels
        mx_t = max(mx_t, S[i].end - S[i].begin);
        mx_c = max(mx_c, S[i].capacity);
    }

    // Graph construction: one vertex per satellite
    int s = n, d = n + 1;
    vector<vector<int>> G(n + 2);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            // handover feasible: S_i -> S_j
            if (S[j].begin <= S[i].end && S[i].end < S[j].end) {
                G[i].push_back(j);
            }
        }

        if (S[i].begin <= T_b && T_b <= S[i].end) {
            G[s].push_back(i);
        }

        if (S[i].begin <= T_e && T_e <= S[i].end) {
            G[i].push_back(d);
        }
    }

    vector<int> blocking_rate(n, 0);
    int mx_handover = 0;
    int sum_handover = 0;
    int mx_blocking = 0;
    int sum_blocking = 0;

    // For each UE, find one shortest path
    for (int user = 0; user < m; user++) {
        vector<double> dist(n + 2, INF);
        vector<int> prev(n + 2, -1);
        vector<bool> vis(n + 2, false);
        priority_queue<pdi, vector<pdi>, greater<pdi>> pq;

        dist[s] = 0.0;
        pq.push({0.0, s});

        while (!pq.empty()) {
            auto [du, u] = pq.top();
            pq.pop();

            if (vis[u]) continue;
            vis[u] = true;

            if (u == d) break;

            for (int v : G[u]) {
                if (vis[v]) continue;

                double w = 0.0;

                if (u == s) {
                    // from source to first satellite:
                    // only consider available channels of first sat
                    double Ca = (load[v] > 0) ? double(load[v]) / max(1, mx_c) : 0.0;
                    w = 0.7 * 0.0 + 0.3 * (1.0 - Ca);
                }
                else if (v == d) {
                    w = 0.0;
                }
                else {
                    // Original GBW idea:
                    // prefer longer remaining visible time + more available channels
                    // Since Dijkstra minimizes weight, convert "larger is better"
                    // into "smaller is better" by using 1 - normalized score.

                    double Tr = double(S[v].end - S[u].end) / max(1, mx_t);
                    if (Tr < 0.0) Tr = 0.0;
                    if (Tr > 1.0) Tr = 1.0;

                    double Ca = (load[v] > 0) ? double(load[v]) / max(1, mx_c) : 0.0;
                    if (Ca < 0.0) Ca = 0.0;
                    if (Ca > 1.0) Ca = 1.0;

                    double score = 0.5 * Tr + 0.5 * Ca;
                    w = 1.0 - score;
                }

                if (dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    pq.push({dist[v], v});
                }
            }
        }

        // Recover path
        vector<int> path_rev;
        int cur = d;
        while (cur != -1) {
            path_rev.push_back(cur);
            cur = prev[cur];
        }
        reverse(path_rev.begin(), path_rev.end());

        // If no valid path
        if (path_rev.empty() || path_rev.front() != s || path_rev.back() != d) {
            continue;
        }

        // Count handovers and update load / blocking
        int cost = -1;
        for (int i = 1; i + 1 < (int)path_rev.size(); i++) {
            int sat = path_rev[i];
            cost++;

            load[sat]--;

            if (load[sat] < 0) {
                int prev_end = 0;
                if (i - 1 >= 1) {
                    prev_end = S[path_rev[i - 1]].end;
                }
                blocking_rate[sat] += S[sat].end - prev_end;
            }
        }

        mx_handover = max(mx_handover, cost);
        sum_handover += cost;
    }

    for (int i = 0; i < n; i++) {
        mx_blocking = max(mx_blocking, blocking_rate[i]);
        sum_blocking += blocking_rate[i];
    }

    cout << mx_handover << ' '
         << fixed << setprecision(2)
         << double(mx_blocking) / double(T_e - T_b + 1) << '\n';

    cout << fixed << setprecision(2)
         << double(sum_handover) / double(m) << ' '
         << double(sum_blocking) / double(T_e - T_b + 1) << '\n';

    return 0;
}