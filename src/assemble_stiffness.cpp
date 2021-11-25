#include <assemble_stiffness.h>

using namespace Eigen;

void insert_block(
    std::vector<Triplet<double>> &triplets,
    int i,
    int j,
    Ref<const Matrix3d> block
) {
    for (int s = 0; s < block.rows(); s++) {
        for (int t = 0; t < block.cols(); t++) {
            triplets.emplace_back(i + s, j + t, block(s, t));
        }
    }
}

void assemble_stiffness(
    Eigen::SparseMatrixd &K,
    Eigen::Ref<const Eigen::VectorXd> q,
    Eigen::Ref<const Eigen::VectorXd> qdot,
    Eigen::Ref<const Eigen::MatrixXd> V,
    Eigen::Ref<const Eigen::MatrixXi> E,
    Eigen::Ref<const Eigen::VectorXd> l0,
    double k
) {
    std::vector<Triplet<double>> triplets;

    for (int ei = 0; ei < E.rows(); ei++) {
        int i = E(ei, 0); int j = E(ei, 1);

        Ref<const Vector3d> q0 = q.segment(3 * i, 3);
        Ref<const Vector3d> q1 = q.segment(3 * j, 3);

        Matrix66d H;
        d2V_spring_particle_particle_dq2(H, q0, q1, l0(ei), k);

        insert_block(triplets, 3 * i, 3 * i, H.topLeftCorner<3, 3>());
        insert_block(triplets, 3 * i, 3 * j, H.topRightCorner<3, 3>());
        insert_block(triplets, 3 * j, 3 * i, H.topRightCorner<3, 3>());
        insert_block(triplets, 3 * j, 3 * j, H.bottomRightCorner<3, 3>());
    }

    K = SparseMatrixd(q.size(), q.size());
    K.setFromTriplets(triplets.begin(), triplets.end());
    K *= -1;
};
