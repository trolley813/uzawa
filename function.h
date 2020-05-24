#pragma once
#include <vector>

using namespace std;

inline double f(double x, double y) {
    return 3 * x + 5 * y;
}

// ||a-b||^2
inline double diff(const vector<double>& a, const vector<double>& b) {
    double r = 0;
    for (int i = 0; i < a.size(); i++)
        r += (a[i] - b[i]) * (a[i] - b[i]);
    return r;
}