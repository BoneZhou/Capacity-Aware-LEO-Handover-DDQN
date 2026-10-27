#include <bits/stdc++.h>
using namespace std;

struct Satellite
{
    int begin, end, capacity;
    Satellite(int begin, int end, int capacity) : begin(begin), end(end), capacity(capacity) {}
};

int main(int argc, char *argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(0);

    string str;
    int n = 1, m = 2;
    int end_time = 3599;
    cin >> str;
    for (auto &ch : str)
        if (ch == ',')
            n++;
    vector<int> duration(n), total_capacity;
    vector<Satellite> S;
    for (int i = 0; i < end_time; i++)
    {
        int cur_capacity = 0;
        for (int j = 0; j < n; j++)
        {
            int cur;
            char tmp;
            cin >> cur;
            if (j < n - 1)
                cin >> tmp;
            if (cur == 1)
            {
                duration[j]++;
                cur_capacity++;
            }
            else if (duration[j] != 0)
            {
                S.push_back(Satellite(i - duration[j], i, 1));
                duration[j] = 0;
            }
        }
        total_capacity.push_back(cur_capacity);
    }
    for (int j = 0; j < n; j++)
        if (duration[j] != 0)
            S.push_back(Satellite(end_time - duration[j], end_time, 1));
    cout << S.size() << ' ' << *min_element(total_capacity.begin(), total_capacity.end()) << '\n';
    cout << "0 " << end_time << '\n';
    for (int i = 0; i < S.size(); i++)
        cout << S[i].begin << ' ' << S[i].end << ' ' << argv[1] << '\n';

    return 0;
}