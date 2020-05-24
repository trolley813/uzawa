#include "function.h"
#include "gauss.h"
#include <cstdio>
#include <mpi.h>

// 0<=x<=a, 0<=y<=b, split by y=c (1 = lower domain, 2 = upper)
constexpr double a = 10, b = 20, c = 10;

// grid point count (n by n)
constexpr int n = 11;

// step size (x and y in both domains)
constexpr double hx1 = a / (n - 1), hx2 = a / (n - 1), hy1 = c / (n - 1),
                 hy2 = (b - c) / (n - 1);

constexpr double alpha = -0.165;

const double epsilon = 1e-4;

inline int idx(int i, int j) { return i * n + j; }

void uzawa_mpi() {
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  vector<double> lambda(n), oldlambda(n);
  int k = 0;
  vector<vector<double>> g1(n * n), g2(n * n);
  for (int i = 0; i < n * n; i++) {
    g1[i].resize(n * n + 1);
    g2[i].resize(n * n + 1);
  }
  vector<double> u1, u2;
  double d = 0;
  if (world_rank == 0) {
    do {
      printf("Iteration %d\n", ++k);
      // fill both matrices
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (j == 0 || j == n - 1) {
            // boundary (left, right)
            g1[idx(i, j)][idx(i, j)] = 1;
            g1[idx(i, j)][n * n] = 0;
            g2[idx(i, j)][idx(i, j)] = 1;
            g2[idx(i, j)][n * n] = 0;
            continue;
          }
          if (i == 0) {
            // boundary (down)
            // 0 for domain 1, -d/dy for domain 2
            g1[idx(i, j)][idx(i, j)] = 1;
            g1[idx(i, j)][n * n] = 0;
            g2[idx(i, j)][idx(i, j)] = 1;
            g2[idx(i, j)][idx(i + 1, j)] = -1;
            g2[idx(i, j)][n * n] = -lambda[j] * hy2;
            continue;
          }
          if (i == n - 1) {
            // boundary (up)
            // 0 for domain 2, +d/dy for domain 1
            g2[idx(i, j)][idx(i, j)] = 1;
            g2[idx(i, j)][n * n] = 0;
            g1[idx(i, j)][idx(i, j)] = 1;
            g1[idx(i, j)][idx(i - 1, j)] = -1;
            g1[idx(i, j)][n * n] = +lambda[j] * hy1;
            continue;
          }
          // all other (internal) points
          g1[idx(i, j)][idx(i, j)] = -2 / (hx1 * hx1) - 2 / (hy1 * hy1);
          g1[idx(i, j)][idx(i, j + 1)] = 1 / (hy1 * hy1);
          g1[idx(i, j)][idx(i, j - 1)] = 1 / (hy1 * hy1);
          g1[idx(i, j)][idx(i - 1, j)] = 1 / (hx1 * hx1);
          g1[idx(i, j)][idx(i + 1, j)] = 1 / (hx1 * hx1);
          g1[idx(i, j)][n * n] = f(i * hx1, j * hy1);
          g2[idx(i, j)][idx(i, j)] = -2 / (hx2 * hx2) - 2 / (hy2 * hy2);
          g2[idx(i, j)][idx(i, j + 1)] = 1 / (hy2 * hy2);
          g2[idx(i, j)][idx(i, j - 1)] = 1 / (hy2 * hy2);
          g2[idx(i, j)][idx(i - 1, j)] = 1 / (hx2 * hx2);
          g2[idx(i, j)][idx(i + 1, j)] = 1 / (hx2 * hx2);
          g2[idx(i, j)][n * n] = f(i * hx2, c + j * hy2);
        }
      }
      u1 = gauss(g1);
      printf("Sending to proc. 2\n");
      MPI_Send(&g2, n * n * (n * n + 1), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      printf("Receiving from proc. 2\n");
      MPI_Recv(&u2, n * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      oldlambda = lambda;
      // assign lambdas
      for (int i = 0; i < n; i++)
        lambda[i] += alpha * (u1[idx(n - 1, i)] - u2[idx(0, i)]);
      d = diff(lambda, oldlambda);
      printf("diff = %lg\n", d);
    }
    while (d > epsilon);
  } else if (world_rank == 1) {
    printf("Receiving from proc. 1\n");
    MPI_Recv(&g2, n * n * (n * n + 1), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    u2 = gauss(g2);
    printf("Sending to proc. 1\n");
    MPI_Send(&u2, n * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Finalize();
}