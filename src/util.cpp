#include "util.h"

#include <Eigen/Eigenvalues>

using namespace Eigen;

// Compute the center of mass of the vertices with position x and masses m
void center_of_mass(Vector3d &r, const VectorXd &x, const VectorXd &m) {
    int n = x.size() / 3;

    assert(m.size() == n);

    r.setZero();
    for (int i = 0; i < n; i++) r += m(i) * x.segment<3>(3 * i);
    r /= m.sum();
}

// Compute the rotational part of the polar decomposition of A
void polar_rotation(Matrix3d &R, const Matrix3d &A) {
    SelfAdjointEigenSolver<Matrix3d> es(A.transpose() * A);

    Matrix3d D = es.eigenvalues().asDiagonal();
    Matrix3d P = es.eigenvectors();

    Matrix3d S = P * D.cwiseSqrt() * P.transpose();
    R = A * S.inverse();
}
