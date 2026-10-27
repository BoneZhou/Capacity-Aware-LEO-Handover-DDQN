#include <bits/stdc++.h>
using namespace std;

struct satellite {
    int begin, end, capacity;
};

static inline bool visible(const satellite& s, int t){
    return s.begin <= t && t < s.end;
}

static vector<vector<int>> solve_window_mvt(
    const vector<satellite>& S,
    int n, int m,
    int win_b, int win_e,
    const vector<int>& init_sat_of_user
){
    int L = win_e - win_b + 1;
    vector<vector<int>> policy(m, vector<int>(L, -1));

    vector<int> load(n);
    for(int i=0;i<n;i++) load[i] = -S[i].capacity;

    for(int u=0; u<m; u++){
        int cur = win_b;

        int initSat = init_sat_of_user[u];
        if(initSat >= 0 && initSat < n && visible(S[initSat], cur)){
            int sat = initSat;
            ++load[sat];
            int until = min(S[sat].end, win_e + 1);
            for(int t=cur; t<until; t++) policy[u][t - win_b] = sat;
            cur = until;
        }

        while(cur <= win_e){
            int best_choice = -1, best_load = INT_MAX;

            for(int i=0;i<n;i++){
                if(visible(S[i], cur)){
                    if(load[i] < best_load){
                        best_load = load[i];
                        best_choice = i;
                    }else if(load[i] == best_load){
                        if(best_choice == -1 || S[i].end > S[best_choice].end){
                            best_choice = i;
                        }
                    }
                }
            }

            if(best_choice == -1) break;

            ++load[best_choice];
            int until = min(S[best_choice].end, win_e + 1);
            for(int t=cur; t<until; t++) policy[u][t - win_b] = best_choice;
            cur = until;
        }
    }

    return policy;
}

static void evaluate_from_policy(
    const vector<satellite>& S,
    int n, int m,
    int T_b, int T_e,
    const vector<vector<int>>& policy_global,
    int& mx_handover,
    double& mx_block_rate,
    double& avg_handover,
    double& avg_block_rate
){
    int T_total = T_e - T_b + 1;

    vector<int> used(n, 0);
    vector<long long> blocking_rate(n, 0);

    mx_handover = 0;
    long long sum_handover = 0;

    for(int u=0; u<m; u++){
        vector<tuple<int,int,int>> segs;

        int last = -2;
        int seg_start = 0;
        for(int t=0; t<T_total; t++){
            int sat = policy_global[t][u];
            if(t==0){
                last = sat;
                seg_start = 0;
            }else if(sat != last){
                segs.emplace_back(last, seg_start, t);
                seg_start = t;
                last = sat;
            }
        }
        segs.emplace_back(last, seg_start, T_total);

        vector<int> sat_segs;
        for(auto &tp: segs){
            int sat, a, b;
            tie(sat,a,b) = tp;
            if(sat >= 0 && sat < n){
                if(sat_segs.empty() || sat_segs.back() != sat) sat_segs.push_back(sat);
            }
        }
        int ho = sat_segs.empty() ? 0 : (int)sat_segs.size() - 1;
        mx_handover = max(mx_handover, ho);
        sum_handover += ho;

        for(auto &tp: segs){
            int sat, a, b;
            tie(sat,a,b) = tp;
            if(!(sat >= 0 && sat < n)) continue;

            int start_time = T_b + a;
            int end_time   = T_b + b;

            used[sat] += 1;
            if(used[sat] > S[sat].capacity){
                long long add = (long long)(end_time - start_time);
                if(add > 0) blocking_rate[sat] += add;
            }
        }
    }

    long long mx_blocking = 0;
    long long sum_blocking = 0;
    for(int i=0;i<n;i++){
        mx_blocking = max(mx_blocking, blocking_rate[i]);
        sum_blocking += blocking_rate[i];
    }

    double denom = double(T_total);
    mx_block_rate = (denom > 0 ? double(mx_blocking)/denom : 0.0);
    avg_handover  = (m > 0 ? double(sum_handover)/double(m) : 0.0);
    avg_block_rate = (denom > 0 ? double(sum_blocking)/denom : 0.0);
}

static void apply_blocking_mask_mac_style(
    const vector<satellite>& S,
    int n, int m,
    int T_b, int T_e,
    vector<vector<int>>& policy_global
){
    int T_total = T_e - T_b + 1;
    vector<int> used(n, 0);

    for(int u = 0; u < m; u++){
        vector<tuple<int,int,int>> segs;

        int last = -2;
        int seg_start = 0;
        for(int t = 0; t < T_total; t++){
            int sat = policy_global[t][u];
            if(t == 0){
                last = sat;
                seg_start = 0;
            }else if(sat != last){
                segs.emplace_back(last, seg_start, t);
                seg_start = t;
                last = sat;
            }
        }
        segs.emplace_back(last, seg_start, T_total);

        for(auto &tp : segs){
            int sat, a, b;
            tie(sat, a, b) = tp;
            if(!(sat >= 0 && sat < n)) continue;

            used[sat] += 1;
            if(used[sat] > S[sat].capacity){
                for(int t = a; t < b; t++) policy_global[t][u] = -1;
            }
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
    for(int i=0;i<n;i++) cin >> S[i].begin >> S[i].end >> S[i].capacity;

    const int W = 900;
    const int H = 60;

    int T_total = T_e - T_b + 1;
    vector<vector<int>> policy_global(T_total, vector<int>(m, -1));
    vector<int> init_sat_of_user(m, -1);

    int first_b = T_b;
    int first_e = min(T_e, first_b + W - 1);
    auto pol0 = solve_window_mvt(S, n, m, first_b, first_e, init_sat_of_user);

    int first_len = first_e - first_b + 1;
    for(int dt=0; dt<first_len; dt++){
        int t = first_b + dt;
        int gi = t - T_b;
        for(int u=0; u<m; u++) policy_global[gi][u] = pol0[u][dt];
    }

    int next_write_t = first_b + first_len;
    if(next_write_t <= T_e){
        for(int u=0; u<m; u++) init_sat_of_user[u] = pol0[u][first_len - 1];
    }

    for(int cur_b = T_b + H; cur_b <= T_e && next_write_t <= T_e; cur_b += H){
        int cur_e = min(T_e, cur_b + W - 1);
        auto pol = solve_window_mvt(S, n, m, cur_b, cur_e, init_sat_of_user);

        int write_from = next_write_t;
        int write_to   = min(T_e, next_write_t + H - 1);

        for(int t = write_from; t <= write_to; t++){
            int gi = t - T_b;
            int local = t - cur_b;
            if(local < 0 || local >= (int)pol[0].size()) continue;
            for(int u=0; u<m; u++) policy_global[gi][u] = pol[u][local];
        }

        int last_t = write_to;
        int last_local = last_t - cur_b;
        if(0 <= last_local && last_local < (int)pol[0].size()){
            for(int u=0; u<m; u++) init_sat_of_user[u] = pol[u][last_local];
        }else{
            for(int u=0; u<m; u++) init_sat_of_user[u] = -1;
        }

        next_write_t += H;
    }

    apply_blocking_mask_mac_style(S, n, m, T_b, T_e, policy_global);

    int mx_handover = 0;
    double mx_block_rate = 0.0, avg_ho = 0.0, avg_block_rate = 0.0;

    evaluate_from_policy(S, n, m, T_b, T_e, policy_global,
                         mx_handover, mx_block_rate, avg_ho, avg_block_rate);

    cout << mx_handover << ' ' << fixed << setprecision(2) << mx_block_rate << "\n";
    cout << fixed << setprecision(2) << avg_ho << ' ' << avg_block_rate << "\n";

    ofstream fout(policy_filename);
    if(!fout){
        cerr << "Failed to open output file: " << policy_filename << "\n";
        return 1;
    }

    fout << "t,user,sat\n";
    for(int u = 0; u < m; u++){
        for(int t = T_b; t <= T_e; t++){
            fout << (t - T_b) << "," << u << "," << policy_global[t - T_b][u] << "\n";
        }
    }
    fout.close();

    cerr << "Saved policy file: " << policy_filename << "\n";
    return 0;
}