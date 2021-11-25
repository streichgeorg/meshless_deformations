#include <assemble_forces.h>
#include <iostream>

using namespace Eigen;

Vector3d g(0.0, 0.0, 0.0);

void assemble_forces(
    Eigen::VectorXd &f,
    Eigen::Ref<const Eigen::VectorXd> q,
    Eigen::Ref<const Eigen::VectorXd> qdot,
    Eigen::Ref<const Eigen::MatrixXd> V,
    Eigen::Ref<const Eigen::MatrixXi> E,
    Eigen::Ref<const Eigen::VectorXd> l0,
    double mass,
    double k
) {
    int n = q.size() / 3;

    f = VectorXd::Zero(3 * n);

    // Gravity
    for (int i = 0; i < n; i++) {
        Vector3d f_gravity;
        dV_gravity_particle_dq(f_gravity, mass, g);
        f.segment(3 * i, 3) -= f_gravity;
    }

    // Springs
    for (int ei = 0; ei < E.rows(); ei++) {
        int i = E(ei, 0); int j = E(ei, 1);

        Vector3d q0 = q.segment(3 * i, 3);
        Vector3d q1 = q.segment(3 * j, 3);

        Vector6d f_spring;
        dV_spring_particle_particle_dq(f_spring, q0, q1, l0(ei), k);

        f.segment(3 * i, 3) -= f_spring.head(3);
        f.segment(3 * j, 3) -= f_spring.tail(3);
    }
};
