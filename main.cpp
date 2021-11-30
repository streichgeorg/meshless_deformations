#include <iostream>
#include <thread>

#include <visualization.h>
#include <igl/edges.h>
#include <igl/edge_lengths.h>
#include <igl/readMESH.h>
#include <igl/boundary_facets.h>
#include <Eigen/Dense>
#include <EigenTypes.h>
#include <Eigen/Eigenvalues>

#include <init_state.h>
#include <dV_spring_particle_particle_dq.h>

using namespace Eigen;

double dt = 1e-4;
double alpha = 1;

double k_selected = 1e7;

MatrixXd V;
MatrixXi T;
MatrixXi F;

VectorXd x;
VectorXd xdot;

VectorXd Xbar;

VectorXd m;

void polar_rotation(Matrix3d &R, const Matrix3d &A) {
    SelfAdjointEigenSolver<Matrix3d> es(A.transpose() * A);

    Matrix3d D = es.eigenvalues().asDiagonal();
    Matrix3d P = es.eigenvectors();

    Matrix3d S = P * D.cwiseSqrt() * P.transpose();
    R = A * S.inverse();
}

void center_of_mass(Vector3d &r, const VectorXd &x, const VectorXd &m) {
    int n = x.size() / 3;

    r.setZero();
    for (int i = 0; i < n; i++) r += m(i) * x.segment<3>(3 * i);
    r /= m.sum();
}

void goal_minimization(
    Matrix3d &R, Vector3d &c,
    const VectorXd &Xbar, const VectorXd &x, const VectorXd &m
) {
    center_of_mass(c, x, m);

    Matrix3d Arot = Matrix3d::Zero();

    int n = Xbar.rows() / 3;

    for (int i = 0; i < n; i++) {
        Arot += m(i) * (x.segment<3>(3 * i) - c) * Xbar.segment<3>(3 * i).transpose();
    }

    polar_rotation(R, Arot);
}

bool simulating = true;

void simulate() {
    int n = Xbar.rows() / 3;

    for(int i = 0; ; i++) {
        Matrix3d R;
        Vector3d c;
        goal_minimization(R, c, Xbar, x, m);

        MatrixXd G = R * Map<MatrixXd>(Xbar.data(), 3, n);
        G.colwise() += c;

        VectorXd f_ext = VectorXd::Zero(x.size());

        for(unsigned int pickedi = 0; pickedi < Visualize::picked_vertices().size(); pickedi++) {
            Vector3d mouse = x.segment<3>(3 * Visualize::picked_vertices()[pickedi]) +
                    Visualize::mouse_drag_world() +
                    Eigen::Vector3d::Constant(1e-6);

            Vector6d dV_mouse;
            dV_spring_particle_particle_dq(
                dV_mouse,
                mouse,
                x.segment<3>(3*Visualize::picked_vertices()[pickedi]), 0.0,
                (Visualize::is_mouse_dragging() ? k_selected : 0.)
            );

            f_ext.segment<3>(3*Visualize::picked_vertices()[pickedi]) -= dV_mouse.segment<3>(3);
        }

        // Add some friction
        f_ext -= 1e2 * xdot;

        VectorXd g = Map<VectorXd>(G.data(), 3 * n);
        VectorXd f_def = alpha * (g - x);

        xdot += f_def / dt + dt * f_ext.cwiseQuotient(m);
        x += dt * xdot;
    }
}

bool draw(igl::opengl::glfw::Viewer & viewer) {

    Visualize::update_vertex_positions(0, x);

    return false;
}

void flatten(VectorXd &x, const MatrixXd &V) {
    MatrixXd VT = V.transpose();
    x = Map<VectorXd>(VT.data(), VT.rows() * VT.cols());
}

int main(int argc, char **argv) {

    std::cout<<"Start Our Project\n";

    // load geometric data
    igl::readMESH("../data/coarse_bunny.mesh", V, T, F);
    igl::boundary_facets(T, F);
    F = F.rowwise().reverse().eval();

    flatten(Xbar, V);

    m = VectorXd::Constant(3 * V.rows(), M_PI);

    Vector3d C;
    center_of_mass(C, Xbar, m);

    flatten(Xbar, V.rowwise() - C.transpose());

    flatten(x, V);
    xdot = VectorXd::Zero(x.size());

    std::thread simulation_thread(simulate);
    simulation_thread.detach();

    //setup libigl viewer and activate
    Visualize::setup(true);
    Visualize::add_object_to_scene(V, F, Eigen::RowVector3d(120, 165, 230) / 255.);
    Visualize::viewer().callback_post_draw = &draw;

    Visualize::viewer().launch_init(true, false, "Meshless Deformations", 0, 0);
    Visualize::viewer().launch_rendering(true);
    simulating = false;
    Visualize::viewer().launch_shut();
}
