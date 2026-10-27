#include <bits/stdc++.h>
using namespace std;

#define MAX INT_MAX

struct satellite
{
    int begin, end, capacity;
};

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, T_b, T_e;
    cin >> n >> m >> T_b >> T_e;

    vector<satellite> S(n);
    vector<int> base_load(n);
    for (int i = 0; i < n; i++)
    {
        cin >> S[i].begin >> S[i].end >> S[i].capacity;
        base_load[i] = -S[i].capacity;
    }

    vector<int> blocking_rate(n, 0);
    int mx = 0, mx_blocking_rate = 0, sum = 0, sum_blocking_rate = 0;

    const int MIN_STAY = 150;   // 至少待 30 秒才能再換

    for (int p = 0; p < m; p++)
    {
        vector<int> load = base_load;

        int cost = 0;
        int prev_choice = -1;
        int lock_until = T_b - 1;   // 在 cur < lock_until 前不能主動換手

        for (int cur = T_b; cur < T_e; cur++)
        {
            int best_choice = -1;
            int best_load = MAX;

            // 找目前時刻最佳衛星
            for (int i = 0; i < n; i++)
            {
                if (S[i].begin <= cur && cur < S[i].end)
                {
                    if (best_choice == -1 || load[i] < best_load)
                    {
                        best_choice = i;
                        best_load = load[i];
                    }
                    else if (load[i] == best_load && S[i].end > S[best_choice].end)
                    {
                        best_choice = i;
                        best_load = load[i];
                    }
                }
            }

            // 如果目前沒有任何可見衛星
            if (best_choice == -1)
            {
                prev_choice = -1;
                continue;
            }

            // 檢查目前連線衛星是否仍可見
            bool prev_visible = false;
            if (prev_choice != -1)
            {
                prev_visible = (S[prev_choice].begin <= cur && cur < S[prev_choice].end);
            }

            // 如果還在 lock 期間，而且目前衛星還可見，就強制留在原衛星
            if (prev_choice != -1 && prev_visible && cur < lock_until)
            {
                best_choice = prev_choice;
            }

            // 若 lock 已過，才允許主動換手
            if (prev_choice != -1 && best_choice != prev_choice)
            {
                cost++;
                lock_until = cur + MIN_STAY;   // 換手後鎖住 30 秒
            }

            // 更新負載
            if (++load[best_choice] > 0)
            {
                blocking_rate[best_choice] += 1;
            }

            prev_choice = best_choice;
        }

        mx = max(mx, cost);
        sum += cost;
    }

    for (int i = 0; i < n; i++)
    {
        mx_blocking_rate = max(mx_blocking_rate, blocking_rate[i]);
        sum_blocking_rate += blocking_rate[i];
    }

    cout << mx << ' ' << fixed << setprecision(2)
         << double(mx_blocking_rate) / double(T_e - T_b + 1) << '\n';

    cout << fixed << setprecision(2)
         << double(sum) / double(m) << ' '
         << double(sum_blocking_rate) / double(T_e - T_b + 1) << '\n';

    return 0;
}