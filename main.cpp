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
double alpha = 5e-1;
double beta = 0.9;

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

    assert(m.size() == n);

    r.setZero();
    for (int i = 0; i < n; i++) r += m(i) * x.segment<3>(3 * i);
    r /= m.sum();
}

void goal_minimization(
    VectorXd &g,
    const VectorXd &Xbar, const VectorXd &x, const VectorXd &m,
    bool quadratic
) {
    int n = Xbar.rows() / 3;

    Vector3d C;
    center_of_mass(C, Xbar, m);

    ArrayXXd Xbarr = (Map<const MatrixXd>(Xbar.data(), 3, n).colwise() - C).array();

    MatrixXd Z(9, n);
    Z << Xbarr,
         Xbarr.pow(2),
         Xbarr.row(0) * Xbarr.row(1), Xbarr.row(1) * Xbarr.row(2), Xbarr.row(0) * Xbarr.row(2);

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

    Matrix<double, 9, 9> A_sc_inv = Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < n; i++) {
        A_sc_inv += m(i) * Z.col(i) * Z.col(i).transpose();
    }

    Matrix<double, 3, 9> A = A_rot * A_sc_inv.inverse();

    Matrix<double, 3, 9> T;
    if (quadratic) T = beta * A + (1 - beta) * R_;
    else T = R_;

    MatrixXd G = T * Z;
    G.colwise() += c;

    g = Map<VectorXd>(G.data(), 3 * n);
}

bool simulating = true;

void simulate() {
    int n = Xbar.rows() / 3;

    for(int t = 0; ; t++) {

        VectorXd f_def = VectorXd::Zero(x.size());

        VectorXi N = VectorXi::Zero(n);
        for (int i = 0; i < T.rows(); i++) {
            for (int j = 0; j < 4; j++) N(T(i, j)) += 1;
        }

        for (int i = 0; i < T.rows(); i++) {
            Matrix<double, 12, 1> XbarT;
            Matrix<double, 12, 1> xT;
            Vector4d mT;

            for (int j = 0; j < 4; j++) {
                XbarT.segment<3>(3 * j) = Xbar.segment<3>(3 * T(i, j));
                xT.segment<3>(3 * j) = x.segment<3>(3 * T(i, j));
                mT(j) = m(T(i, j));
            }

            VectorXd g;
            goal_minimization(g, XbarT, xT, mT, false);

            for (int j = 0; j < 4; j++) {
                f_def.segment<3>(3 * T(i, j)) += alpha * (g - xT).segment<3>(3 * j) / N(T(i, j));
            }
        }

        VectorXd g;
        goal_minimization(g, Xbar, x, m, true);
        f_def += 1e-2 * alpha * (g - x);

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

        f_ext -= 1e3 * xdot;

        VectorXd v_ext(3 * n);
        for (int i = 0; i < n; i++) v_ext.segment<3>(3 * i) = f_ext.segment<3>(3 * i) / m(i);

        xdot += f_def / dt + dt * v_ext;
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

    m = VectorXd::Constant(V.rows(), M_PI);

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
