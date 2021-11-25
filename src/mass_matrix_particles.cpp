#include <mass_matrix_particles.h>

#include <iostream>

#include <vector>

using namespace Eigen;

void mass_matrix_particles(
    Eigen::SparseMatrixd &M,
    Eigen::Ref<const Eigen::VectorXd> q,
    double mass
) {
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < q.size(); i++) triplets.emplace_back(i, i, mass);
    M = SparseMatrixd(q.size(), q.size());
    M.setFromTriplets(triplets.begin(), triplets.end());
}
