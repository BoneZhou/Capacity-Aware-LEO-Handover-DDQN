#include <bits/stdc++.h>

/*
這個程式是直接拿 MSH 的圖直接跑 m 次 dijkstra，這樣就可以達到 MVT 的目標。
實作可以參考 MSH
*/

using namespace std;
#define MAX INT_MAX
#define pii pair<int, int>
struct satellite
{
    int begin, end, capacity;
};

struct node
{
    int next, prev;
    node() { next = prev = -1; };
};

int main()
{
    // Input
    int n, m, T_b, T_e;
    cin >> n >> m >> T_b >> T_e;
    vector<satellite> S(n);
    for (int i = 0; i < n; i++)
        cin >> S[i].begin >> S[i].end >> S[i].capacity;

    // constuction of G(V, E)
    int s, d;
    vector<int> cnt(n);
    vector<pii> table;
    vector<vector<pii>> G;
    // create vertex
    s = 0;
    table.emplace_back(0, 0);
    G.emplace_back(vector<pii>());
    for (int i = 0; i < n; i++)
    {
        cnt[i] = G.size();
        for (int c = 0; c < S[i].capacity; c++)
        {
            table.emplace_back(i + 1, c + 1);
            G.emplace_back(vector<pii>());
            G[G.size() - 1].emplace_back(G.size(), 0);
            table.emplace_back(i + 1, c + 1);
            G.emplace_back(vector<pii>());
        }
    }
    cnt.emplace_back(G.size());
    d = G.size();
    table.emplace_back(0, 0);
    G.emplace_back(vector<pii>());

    // create edge
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                continue;
            if (S[j].begin <= S[i].end && S[i].end <= S[j].end)
                for (int k = cnt[i] + 1; k < cnt[i + 1]; k += 2)
                    for (int l = cnt[j]; l < cnt[j + 1]; l += 2)
                        G[k].emplace_back(l, 1);
        }

        if (S[i].begin <= T_b && T_b <= S[i].end)
            for (int k = cnt[i]; k < cnt[i + 1]; k += 2)
                G[s].emplace_back(k, 0);

        if (S[i].begin <= T_e && T_e <= S[i].end)
            for (int k = cnt[i] + 1; k < cnt[i + 1]; k += 2)
                G[k].emplace_back(d, 0);
    }

    // evaluation of Pj
    vector<node> node(G.size());

    vector<int> dist(G.size(), MAX), p(G.size());
    vector<bool> visited(G.size());
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    pq.emplace(0, 0);
    dist[s] = 0;
    while (pq.size())
    {
        int u = pq.top().second;
        pq.pop();
        if (visited[u])
            continue;
        visited[u] = true;
        if (u == d)
            break;
        for (auto &[v, w] : G[u])
        {
            if (dist[v] > dist[u] + w)
            {
                dist[v] = dist[u] + w;
                p[v] = u;
                visited[v] = false;
                pq.emplace(dist[v], v);
            }
        }
    }
    int cur = d, prev = p[d];
    while (cur != s)
    {
        node[prev].next = cur;
        cur = prev;
        prev = p[cur];
    }

    // print result
    int tmp = -1;
    vector<int> blocking_rate(n);
    int mx_blocking_rate = 0, sum_blocking_rate = 0;
    cur = s;
    while (cur != d)
    {
        if (cur & 1)
        {
            tmp++;
        }
        else
        {
            int u = table[cur].first - 1, v = table[node[cur].next].first - 1;
            if (u == -1)
                blocking_rate[v] = (S[v].end - T_b) * max(0, (m - S[v].capacity));
            else if (v != -1)
                blocking_rate[v] = (S[v].end - S[u].end) * max(0, (m - S[v].capacity));
        }
        cur = node[cur].next;
    }
    for (int i = 0; i < n; i++)
    {
        mx_blocking_rate = max(blocking_rate[i], mx_blocking_rate);
        sum_blocking_rate += blocking_rate[i];
    }

    cout << tmp << ' ' << fixed << setprecision(2) << double(mx_blocking_rate) / double(T_e - T_b + 1) << '\n';
    cout << tmp << ' ' << fixed << setprecision(2) << double(sum_blocking_rate) / double(T_e - T_b + 1) << '\n';

    return 0;
}