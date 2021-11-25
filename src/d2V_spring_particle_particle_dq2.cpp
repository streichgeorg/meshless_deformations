#include <d2V_spring_particle_particle_dq2.h>

using namespace Eigen;

// I computed the derivative with matrixcalculus.org
// Not sure if thats allowed or not, but on paper it got pretty messy :/
void d2V_spring_particle_particle_dq2(
    Eigen::Ref<Eigen::Matrix66d> H,
    Eigen::Ref<const Eigen::Vector3d> q0,
    Eigen::Ref<const Eigen::Vector3d> q1,
    double l0,
    double k
) {
    Vector3d t0 = q0 - q1;
    double t1 = t0.norm();
    Matrix3d T2 = t0 * t0.transpose();
    double t3 = k * (t1 - l0);

    Matrix3d G = k / std::pow(t1, 2) * T2 -
                 t3 / std::pow(t1, 3) * T2 +
                 t3 / t1 * Matrix3d::Identity();

    H << G, -G,
         -G, G;
}
