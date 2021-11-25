#include <fixed_point_constraints.h>
#include <algorithm>

#include <iostream>

using namespace Eigen;

void fixed_point_constraints(Eigen::SparseMatrixd &P, unsigned int q_size, const std::vector<unsigned int> indices) {
    int n = q_size / 3;

    std::vector<Triplet<double>> triplets;

    int i = 0;
    for (int j = 0; j < n; j++) {
        bool fixed = false;
        for (int k : indices) {
            if (j == k) {
                fixed = true;
                break;
            }
        }
        if (fixed) continue;

        for (int k = 0; k < 3; k++) triplets.emplace_back(3 * i + k, 3 * j + k, 1);

        i++;
    }

    P = SparseMatrixd(3 * i, 3 * n);
    P.setFromTriplets(triplets.begin(), triplets.end());
}
