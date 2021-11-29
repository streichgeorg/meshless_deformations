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

using namespace Eigen;

double alpha = 0.5;

double dt = 1e-4;

MatrixXd V; //vertices of simulation mesh
MatrixXi T; //faces of simulation mesh
MatrixXi F; //faces of simulation mesh

VectorXd x;
VectorXd xdot;

VectorXd Xbar;

VectorXd m;

void polar_rotation(Matrix3d &R, const Matrix3d &A) {
    SelfAdjointEigenSolver<Matrix3d> es(A.transpose() * A);

    Matrix3d D = es.eigenvalues().asDiagonal();
    Matrix3d P = es.eigenvectors();

    Matrix3d S = P * D.cwiseSqrt() * P.transpose();
    R = S.lu().solve(A);
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

    Matrix3d Arot;

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

        VectorXd g = Map<VectorXd>(G.data(), 3 * n);
        VectorXd f_def = alpha * (g - x);

        xdot += f_def / dt;
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
    igl::readMESH("../data/cube.mesh", V, T, F);
    igl::boundary_facets(T, F);
    F = F.rowwise().reverse().eval();

    m = VectorXd::Constant(V.rows(), M_PI);

    flatten(Xbar, V);

    Vector3d C;
    center_of_mass(C, Xbar, m);

    flatten(Xbar, V.rowwise() - C.transpose());

    flatten(x, V);
    xdot = VectorXd::Zero(x.size());

    std::thread simulation_thread(simulate);
    simulation_thread.detach();

    //setup libigl viewer and activate
    Visualize::setup(true);
    Visualize::add_object_to_scene(V, F, Eigen::RowVector3d(244,165,130)/255.);
    Visualize::viewer().callback_post_draw = &draw;

    Visualize::viewer().launch_init(true, false, "Meshless Deformations", 0, 0);
    Visualize::viewer().launch_rendering(true);
    simulating = false;
    Visualize::viewer().launch_shut();
}
