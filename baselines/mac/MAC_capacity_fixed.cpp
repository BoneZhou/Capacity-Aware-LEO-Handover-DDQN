#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace std;

struct Satellite {
    int begin;
    int end;
    int capacity;
};

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const uint32_t seed = argc > 1 ? static_cast<uint32_t>(stoul(argv[1])) : 42U;
    int n, m, time_begin, time_end;
    if (!(cin >> n >> m >> time_begin >> time_end)) return 1;

    vector<Satellite> satellites(n);
    for (auto& satellite : satellites) {
        cin >> satellite.begin >> satellite.end >> satellite.capacity;
        if (satellite.capacity < 0) throw runtime_error("Negative satellite capacity");
    }

    constexpr int MIN_STAY = 150;
    const int time_slots = time_end - time_begin;
    if (time_slots != 3599) throw runtime_error("Expected exactly 3599 evaluated time slots");

    vector<int> occupancy(n, 0);
    vector<int> connection(m, -1);
    vector<int> lock_until(m, time_begin - 1);
    vector<int> order(m);
    iota(order.begin(), order.end(), 0);
    mt19937 rng(seed);

    long long served = 0;
    long long blocked = 0;
    long long handovers = 0;
    long long failed_handover_attempts = 0;
    int max_occupancy = 0;

    auto visible = [&](int satellite, int time) {
        return satellites[satellite].begin <= time && time < satellites[satellite].end;
    };

    for (int time = time_begin; time < time_end; ++time) {
        for (int user = 0; user < m; ++user) {
            const int current = connection[user];
            if (current != -1 && !visible(current, time)) {
                if (--occupancy[current] < 0) throw runtime_error("Negative occupancy after release");
                connection[user] = -1;
            }
        }

        iota(order.begin(), order.end(), 0);
        shuffle(order.begin(), order.end(), rng);
        int served_this_slot = 0;
        int blocked_this_slot = 0;

        for (const int user : order) {
            const int current = connection[user];
            int target = -1;
            int best_available = -1;

            for (int satellite = 0; satellite < n; ++satellite) {
                if (!visible(satellite, time)) continue;
                const int available = satellites[satellite].capacity - occupancy[satellite];
                if (target == -1 || available > best_available ||
                    (available == best_available && satellites[satellite].end > satellites[target].end)) {
                    target = satellite;
                    best_available = available;
                }
            }

            if (current != -1 && time < lock_until[user]) target = current;

            bool is_served = false;
            if (target == current && current != -1) {
                is_served = true;
            } else if (target != -1 && occupancy[target] < satellites[target].capacity) {
                ++occupancy[target];
                if (current != -1) {
                    if (--occupancy[current] < 0) throw runtime_error("Negative occupancy after handover");
                    ++handovers;
                    lock_until[user] = time + MIN_STAY;
                }
                connection[user] = target;
                is_served = true;
            } else if (current != -1) {
                // A failed optional handover does not tear down an otherwise valid serving link.
                ++failed_handover_attempts;
                is_served = true;
            }

            if (is_served) {
                ++served;
                ++served_this_slot;
            } else {
                connection[user] = -1;
                ++blocked;
                ++blocked_this_slot;
            }

            for (int satellite = 0; satellite < n; ++satellite) {
                if (occupancy[satellite] < 0 || occupancy[satellite] > satellites[satellite].capacity)
                    throw runtime_error("Capacity invariant violated");
                max_occupancy = max(max_occupancy, occupancy[satellite]);
            }
        }

        vector<int> connected_count(n, 0);
        for (const int satellite : connection) {
            if (satellite != -1) {
                if (!visible(satellite, time)) throw runtime_error("Invisible active connection");
                ++connected_count[satellite];
            }
        }
        if (connected_count != occupancy) throw runtime_error("Connection/occupancy mismatch");
        if (served_this_slot + blocked_this_slot != m) throw runtime_error("Per-slot accounting mismatch");
    }

    const long long evaluated = static_cast<long long>(m) * time_slots;
    if (served + blocked != evaluated) throw runtime_error("Full-experiment accounting mismatch");
    const double blocking_percent = 100.0 * static_cast<double>(blocked) / static_cast<double>(evaluated);
    if (blocking_percent < 0.0 || blocking_percent > 100.0) throw runtime_error("Invalid blocking rate");

    cout << seed << ',' << m << ',' << satellites.front().capacity << ',' << time_slots << ','
         << evaluated << ',' << served << ',' << blocked << ',' << setprecision(15) << blocking_percent << ','
         << handovers << ',' << failed_handover_attempts << ',' << max_occupancy << ",0,0\n";
    return 0;
}
