#include "function.h"
#include "gauss.h"

// 0<=x<=a, 0<=y<=b, split by y=c (1 = lower domain, 2 = upper)
constexpr double a = 10, b = 20, c = 10;

// grid point count (n by n)
constexpr int n = 51;

// step size (x and y in both domains)
constexpr double hx1 = a / (n - 1), hx2 = a / (n - 1), hy1 = c / (n - 1),
                 hy2 = (b - c) / (n - 1);

constexpr double alpha = 0.1;

const double epsilon = 1e-4;

inline int idx(int i, int j) { return i * n + j; }

// ||a-b||^2
double diff(const vector<double> &a, const vector<double> &b) {
  double r = 0;
  for (int i = 0; i < a.size(); i++)
    r += (a[i] - b[i]) * (a[i] - b[i]);
  return r;
}

void uzawa() {
  vector<double> lambda(n), oldlambda(n);
  int k = 0;
  vector<vector<double>> g1(n * n), g2(n * n);
  for (int i = 0; i < n; i++) {
      g1[i].resize(n * n + 1);
      g2[i].resize(n * n + 1);
  }
  vector<double> u1, u2;
  do {
      // fill both matrices
      for (int i = 0; i < n * n; i++) {
          
      }
      // solve linear equation system (in parallel)
      u1 = gauss(g1);
      u2 = gauss(g2);
      // assign lambdas
      // lambda += alpha * ...
      oldlambda = lambda;
  } while (diff(lambda, oldlambda) < epsilon);
}