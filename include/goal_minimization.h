#include <Eigen/Dense>

// Perform the goal minimization task as described in the report
void goal_minimization(
    Eigen::VectorXd &g,
    const Eigen::MatrixXd &Z, const Eigen::Matrix<double, 9, 9> &B, const Eigen::VectorXd &m,
    const Eigen::VectorXd &x, bool quadratic
);
