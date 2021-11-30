#include <dV_spring_particle_particle_dq.h>

using namespace Eigen;

// Derivative computed using http://www.matrixcalculus.org
void dV_spring_particle_particle_dq(
    Eigen::Ref<Eigen::Vector6d> f,
    Eigen::Ref<const Eigen::Vector3d> q0,
    Eigen::Ref<const Eigen::Vector3d> q1,
    double l0,
    double k
) {
    double l = (q1 - q0).norm();
    Vector3d e = k * (l - l0) * (q0 - q1) / l;
    f << e, -e;
}
