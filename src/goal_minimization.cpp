#include "goal_minimization.h"

#include "params.h"
#include "util.h"

using namespace Eigen;

void goal_minimization(
    Eigen::VectorXd &g,
    const Eigen::MatrixXd &Z, const Eigen::Matrix<double, 9, 9> &B, const VectorXd &m,
    const Eigen::VectorXd &x, bool quadratic
) {
    int n = x.rows() / 3;

    Vector3d c;
    center_of_mass(c, x, m);

    Matrix<double, 3, 9> A_rot = Matrix<double, 3, 9>::Zero();
    for (int i = 0; i < n; i++) {
        A_rot += m(i) * (x.segment<3>(3 * i) - c) * Z.col(i).transpose();
    }

    Matrix3d R;
    polar_rotation(R, A_rot.leftCols<3>());

    Matrix<double, 3, 9> R_ = Matrix<double, 3, 9>::Zero();
    R_.leftCols<3>() = R;

    Matrix<double, 3, 9> A = A_rot * B;

    Matrix<double, 3, 9> T;
    if (quadratic) T = beta * A + (1 - beta) * R_;
    else T = R_;

    MatrixXd G = T * Z;
    G.colwise() += c;

    g = Map<VectorXd>(G.data(), 3 * n);
}
