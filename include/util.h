#include <Eigen/Dense>

// Compute the center of mass of the vertices with position x and masses m
void center_of_mass(Eigen::Vector3d &r, const Eigen::VectorXd &x, const Eigen::VectorXd &m);

// Compute the rotational part of the polar decomposition of A
void polar_rotation(Eigen::Matrix3d &R, const Eigen::Matrix3d &A);

