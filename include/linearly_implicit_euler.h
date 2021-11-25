#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <EigenTypes.h>

#include <iostream>

using namespace Eigen;

//Input:
//  q - generalized coordiantes for the mass-spring system
//  qdot - generalized velocity for the mass spring system
//  dt - the time step in seconds
//  mass - the mass matrix
//  force(q, qdot) - a function that computes the force acting on the mass-spring system. This takes q and qdot as parameters.
//  stiffness(q, qdot) - a function that computes the stiffness (negative second derivative of the potential energy). This takes q and qdot as parameters.
//  tmp_force - scratch space to collect forces
//  tmp_stiffness - scratch space to collect stiffness matrix
//Output:
//  q - set q to the updated generalized coordinate using linearly implicit time integration
//  qdot - set qdot to the updated generalized velocity using linearly implicit time integration
template<typename FORCE, typename STIFFNESS>
inline void linearly_implicit_euler(
    Eigen::VectorXd &q,
    Eigen::VectorXd &qdot,
    double dt,
    const Eigen::SparseMatrixd &mass,
    FORCE &force,
    STIFFNESS &stiffness,
    Eigen::VectorXd &tmp_force,
    Eigen::SparseMatrixd &tmp_stiffness
) {
    force(tmp_force, q, qdot);
    stiffness(tmp_stiffness, q, qdot);

    SimplicialLDLT<SparseMatrixd> solver;
    solver.compute(mass - std::pow(dt, 2) * tmp_stiffness);

    if (solver.info() != Success) {
        std::cout << "Solver failed :(" << std::endl;
        exit(1);
    }

    qdot = solver.solve(mass * qdot + dt * tmp_force);
    q += dt * qdot;
}
