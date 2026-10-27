#include <bits/stdc++.h>
#include <filesystem>
#include <regex>
using namespace std;
namespace fs = std::filesystem;

static constexpr int kTimeBegin = 0;
static constexpr int kTimeEndExclusive = 3599;
static constexpr int kActionCount = 50;
static constexpr array<int, kActionCount> kActionToRateId = {
    36, 27, 18, 10, 2, 44, 35, 26, 17, 9,
    1, 43, 34, 25, 16, 8, 0, 42, 33, 24,
    15, 7, 49, 41, 32, 23, 14, 6, 48, 40,
    31, 22, 13, 5, 47, 39, 30, 21, 12, 4,
    46, 38, 29, 3, 11, 19, 20, 28, 37, 45
};

// -----------------------------
// 讀 input_angle.txt：key=(sid,t) -> rate
// -----------------------------
struct RateTable {
    unordered_map<long long,double> rt;
    unordered_set<int> satellite_ids;
    int min_sid = INT_MAX, max_sid = INT_MIN;
    int min_t   = INT_MAX, max_t   = INT_MIN;

    static long long key(int sid, int t) {
        return ((long long)sid << 32) | (unsigned int)t;
    }

    bool load(const string& path) {
        ifstream fin(path);
        if(!fin) return false;

        int sid, t;
        double r;
        while(fin >> sid >> t >> r) {
            rt[key(sid, t)] = r;
            satellite_ids.insert(sid);
            min_sid = min(min_sid, sid);
            max_sid = max(max_sid, sid);
            min_t   = min(min_t, t);
            max_t   = max(max_t, t);
        }

        cerr << "[RATE] sid=[" << min_sid << "," << max_sid
             << "], t=[" << min_t << "," << max_t << "]\n";

        return !rt.empty();
    }

    inline double get(int sid, int t) const {
        auto it = rt.find(key(sid, t));
        return (it == rt.end() ? 0.0 : it->second);
    }

    inline bool has_satellite(int sid) const {
        return satellite_ids.find(sid) != satellite_ids.end();
    }
};

static bool validate_rate_mapping(const RateTable& rates, string& error_msg) {
    array<bool, kActionCount> seen{};
    for(int action_id = 0; action_id < kActionCount; ++action_id){
        int rate_id = kActionToRateId[action_id];
        if(rate_id < 0 || rate_id >= kActionCount || seen[rate_id]){
            error_msg = "Action-to-rate mapping is not a permutation of Satellite IDs 0..49";
            return false;
        }
        if(!rates.has_satellite(rate_id)){
            error_msg = "Rate table has no rows for mapped Satellite ID " + to_string(rate_id);
            return false;
        }
        seen[rate_id] = true;
    }
    return true;
}

// -----------------------------
// throughput 計算結果
// -----------------------------
struct ThroughputResult {
    double avg_all = 0.0;
    int m = 0;
    int L = 0;
    long long hits = 0;
    long long misses = 0;
    long long blocks = 0;
    vector<double> avg_user;
    bool ok = false;
    string error_msg;
};

// -----------------------------
// 從單一 policy csv 計算 throughput
// policy 格式：t,user,sat
// -----------------------------
static ThroughputResult calc_throughput_from_policy(
    const RateTable& rates,
    const string& policy_file
){
    ThroughputResult res;

    ifstream fin(policy_file);
    if(!fin){
        res.error_msg = "Failed to open policy file";
        return res;
    }

    string line;
    getline(fin, line); // skip header

    vector<tuple<int,int,int>> rows;
    int max_user = -1;
    int max_t    = -1;

    while(getline(fin, line)){
        if(line.empty()) continue;

        replace(line.begin(), line.end(), ',', ' ');
        istringstream iss(line);

        int t, u, s;
        if(!(iss >> t >> u >> s)) continue;

        if(t < kTimeBegin || t >= kTimeEndExclusive) continue;
        rows.emplace_back(t, u, s);
        max_user = max(max_user, u);
        max_t    = max(max_t, t);
    }

    if(rows.empty()){
        res.error_msg = "No valid rows in policy file";
        return res;
    }

    int m = max_user + 1;
    int L = kTimeEndExclusive - kTimeBegin;

    vector<double> sum_rate(m, 0.0);

    long long hits = 0, misses = 0, blocks = 0;

    for(const auto &row : rows){
        int t, u, s;
        tie(t, u, s) = row;

        double R = 0.0;
        if(s == -1){
            blocks++;
        }else if(s >= 0 && s < kActionCount){
            R = rates.get(kActionToRateId[s], t);
            if(R > 0.0) hits++;
            else misses++;
        }else{
            res.error_msg = "Policy contains unmappable satellite/action ID " + to_string(s)
                          + "; expected BLOCK=-1 or action ID 0..49";
            return res;
        }
        if(u >= 0 && u < m) sum_rate[u] += R;
    }

    vector<double> avg(m, 0.0);
    double avg_all = 0.0;
    for(int u = 0; u < m; ++u){
        avg[u] = sum_rate[u] / double(L);
        avg_all += avg[u];
    }
    avg_all /= double(m);

    res.avg_all = avg_all;
    res.m = m;
    res.L = L;
    res.hits = hits;
    res.misses = misses;
    res.blocks = blocks;
    res.avg_user = std::move(avg);
    res.ok = true;
    return res;
}

// -----------------------------
// 從檔名抓 m 與 c，例如：MSHBO_m25_c3_policy.csv
// 抓不到就回傳 -1, -1
// -----------------------------
static pair<int,int> parse_m_c_from_filename(const string& filename){
    regex pat(R"(.*_m(\d+)_c(\d+)_.*\.csv$)", regex::icase);
    smatch match;
    if(regex_match(filename, match, pat)){
        int m = stoi(match[1].str());
        int c = stoi(match[2].str());
        return {m, c};
    }
    return {-1, -1};
}

// -----------------------------
// 主程式
// 用法：
//   throughput_calculate.exe input_angle.txt root_folder
//
// 例：
//   throughput_calculate input_angle.txt path/to/policy_root
// -----------------------------
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if(argc < 3){
        cerr << "Usage: " << argv[0]
             << " <input_angle.txt> <root_folder> [output_csv] [capacity_filter]\n";
        cerr << "Example: " << argv[0]
             << " input_angle.txt path/to/policy_root"
             << " corrected.csv 5\n";
        return 1;
    }

    string rate_file = argv[1];
    string root_folder = argv[2];

    RateTable rates;
    if(!rates.load(rate_file)){
        cerr << "Failed to load " << rate_file << "\n";
        return 1;
    }

    string mapping_error;
    if(!validate_rate_mapping(rates, mapping_error)){
        cerr << mapping_error << "\n";
        return 1;
    }

    if(!fs::exists(root_folder) || !fs::is_directory(root_folder)){
        cerr << "Root folder does not exist or is not a directory: " << root_folder << "\n";
        return 1;
    }

    string output_csv = argc >= 4
        ? argv[3]
        : (fs::path(root_folder) / "result_throughput.csv").string();
    int capacity_filter = -1;
    if(argc >= 5){
        try {
            capacity_filter = stoi(argv[4]);
        } catch(const exception&) {
            cerr << "capacity_filter must be an integer\n";
            return 1;
        }
    }
    ofstream fout(output_csv);
    if(!fout){
        cerr << "Failed to create " << output_csv << "\n";
        return 1;
    }

    fout << "algorithm,policy_file,m,c,avg_throughput_all_users_mbps,num_users,total_time,hits,misses,blocks\n";

    int total_files = 0;
    int success_files = 0;

    // 掃描 root_folder 底下每個「演算法資料夾」
    for(const auto& entry : fs::directory_iterator(root_folder)){
        if(!entry.is_directory()) continue;

        string algo_name = entry.path().filename().string();
        fs::path algo_dir = entry.path();

        cerr << "\n[SCAN] algorithm folder: " << algo_name << "\n";

        vector<fs::path> csv_files;
        for(const auto& f : fs::directory_iterator(algo_dir)){
            if(!f.is_regular_file()) continue;

            string ext = f.path().extension().string();
            string fname = f.path().filename().string();

            if(ext == ".csv" || ext == ".CSV"){
                // 避免把 throughput 結果表再讀回去
                if(fname == "result_throughput.csv") continue;
                csv_files.push_back(f.path());
            }
        }

        sort(csv_files.begin(), csv_files.end());

        for(const auto& policy_path : csv_files){
            string policy_file = policy_path.filename().string();
            auto [m_from_name, c_from_name] = parse_m_c_from_filename(policy_file);
            if(capacity_filter >= 0 && c_from_name != capacity_filter) continue;
            total_files++;

            cerr << "[PROCESS] " << policy_path.string() << "\n";

            ThroughputResult res = calc_throughput_from_policy(rates, policy_path.string());
            if(!res.ok){
                cerr << "  -> failed: " << res.error_msg << "\n";
                fout << algo_name << ","
                     << policy_file << ","
                     << m_from_name << ","
                     << c_from_name << ","
                     << "ERROR,ERROR,ERROR,ERROR,ERROR,ERROR\n";
                continue;
            }

            success_files++;

            fout << algo_name << ","
                 << policy_file << ","
                 << m_from_name << ","
                 << c_from_name << ","
                 << fixed << setprecision(6) << res.avg_all << ","
                 << res.m << ","
                 << res.L << ","
                 << res.hits << ","
                 << res.misses << ","
                 << res.blocks << "\n";

            cerr << "  -> avg throughput = " << fixed << setprecision(6)
                 << res.avg_all << " Mbps\n";
        }
    }

    fout.close();

    cerr << "\nDone.\n";
    cerr << "Output CSV: " << output_csv << "\n";
    cerr << "Processed files: " << total_files
         << ", success: " << success_files << "\n";

    return 0;
}
