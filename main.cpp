#include <iostream>
#include <thread>

#include <visualization.h>
#include <igl/edges.h>
#include <igl/edge_lengths.h>
#include <igl/readMESH.h>
#include <igl/boundary_facets.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <EigenTypes.h>
#include <Eigen/Eigenvalues>

#include <init_state.h>
#include <dV_spring_particle_particle_dq.h>

#include <set>

using namespace Eigen;

void center_of_mass(Vector3d &r, const VectorXd &x, const VectorXd &m) {
    int n = x.size() / 3;

    assert(m.size() == n);

    r.setZero();
    for (int i = 0; i < n; i++) r += m(i) * x.segment<3>(3 * i);
    r /= m.sum();
}


struct Cluster {
    std::vector<int> vertices;

    VectorXd m;
    MatrixXd Z;
    Matrix<double, 9, 9> B;

    Cluster(const std::set<int>& vertex_set, const MatrixXd &V) {
        std::copy(vertex_set.begin(), vertex_set.end(), std::back_inserter(vertices));

        int n = vertices.size();

        VectorXd X = VectorXd::Zero(3 * n);

        for (int i = 0; i < n; i++) {
            X.segment<3>(3 * i) = V.row(vertices[i]).transpose();
            
        }

        m = VectorXd::Constant(vertices.size(), M_PI);

        Vector3d C;
        center_of_mass(C, X, m);

        ArrayXXd Xbarr = (Map<const MatrixXd>(X.data(), 3, n).colwise() - C).array();

        Z = MatrixXd(9, n);
        Z << Xbarr,
             Xbarr.pow(2),
             Xbarr.row(0) * Xbarr.row(1), Xbarr.row(1) * Xbarr.row(2), Xbarr.row(0) * Xbarr.row(2);

        Matrix<double, 9, 9> B_inv = Matrix<double, 9, 9>::Zero();
        for (int i = 0; i < n; i++) {
            B += m(i) * Z.col(i) * Z.col(i).transpose();
        }
        B = B.inverse();
    }
};


double dt = 1e-4;
double alpha = 0.2;
double beta = 0.0;

double k_selected = 1e8;

MatrixXd V;
MatrixXi T;
MatrixXi F;

VectorXd x;
VectorXd xdot;

VectorXd m;

std::vector<Cluster> clusters;

void polar_rotation(Matrix3d &R, const Matrix3d &A) {
    SelfAdjointEigenSolver<Matrix3d> es(A.transpose() * A);

    Matrix3d D = es.eigenvalues().asDiagonal();
    Matrix3d P = es.eigenvectors();

    Matrix3d S = P * D.cwiseSqrt() * P.transpose();
    R = A * S.inverse();
}

void goal_minimization(
    VectorXd &g,
    const MatrixXd &Z, const Matrix<double, 9, 9> B, const VectorXd &m,
    const VectorXd &x, bool quadratic
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

bool simulating = true;

void simulate() {
    int n = x.rows() / 3;

    for(int t = 0; ; t++) {

        VectorXd f_def = VectorXd::Zero(x.size());

        VectorXd N = VectorXd::Zero(x.size() / 3);
        for (auto &cluster : clusters) {
            for (int k : cluster.vertices) N(k) += 1;
        }

        for (auto &cluster : clusters) {
            VectorXd x_cluster(3 * cluster.vertices.size());
            for (int i = 0; i < cluster.vertices.size(); i++) {
                x_cluster.segment<3>(3 * i) = x.segment<3>(3 * cluster.vertices[i]);
            }

            VectorXd g;
            goal_minimization(g, cluster.Z, cluster.B, cluster.m, x_cluster, true);

            for (int i = 0; i < cluster.vertices.size(); i++) {
                VectorXd d = g - x_cluster;
                f_def.segment<3>(3 * cluster.vertices[i]) += alpha * d.segment<3>(3 * i) / N(cluster.vertices[i]);
            }
        }

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

std::vector<Cluster> clusters_from_file(const MatrixXi &T, const MatrixXd &V, VectorXd &c) {
    std::ifstream f("../data/coarse_bunny.clusters");

    if (!f.is_open()) {
        std::cout << "Failed to read clusters file" << std::endl;
        exit(-1);
    }

    c = VectorXd(V.rows());

    std::vector<int> idx_to_cluster;

    int n_clusters = 0;
    int cluster;
    while (f >> cluster) {
        idx_to_cluster.push_back(cluster);
        n_clusters = std::max(n_clusters, cluster + 1);
    }

    for (int i = 0; i < V.rows(); i++) c(i) = (double) idx_to_cluster[i] / n_clusters;

    f.close();

    if (!idx_to_cluster.size() == V.rows()) {
        std::cout << "Clusters has wrong format" << std::endl;
    }

    std::vector<std::set<int>> cluster_vertices(n_clusters, std::set<int>());
    for (int i = 0; i < T.rows(); i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                cluster_vertices[idx_to_cluster[T(i, j)]].insert(T(i, k));
            }
        }
    }

    std::vector<Cluster> clusters;
    for (int i = 0; i < n_clusters; i++) clusters.emplace_back(cluster_vertices[i], V);

    return clusters;
}

bool key_down_callback(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers) {

    if(key == 'Q') {
        std::cout << "test" << std::endl;
    }

    return false;
}

int main(int argc, char **argv) {

    std::cout<<"Start Our Project\n";

    // load geometric data
    igl::readMESH("../data/coarse_bunny.mesh", V, T, F);
    igl::boundary_facets(T, F);
    F = F.rowwise().reverse().eval();

    VectorXd c;
    clusters = clusters_from_file(T, V, c);

    m = VectorXd::Constant(V.rows(), M_PI);

    flatten(x, V);
    xdot = VectorXd::Zero(x.size());

    std::thread simulation_thread(simulate);
    simulation_thread.detach();

    //setup libigl viewer and activate
    Visualize::setup(true);
    Visualize::add_object_to_scene(V, F, c);
    Visualize::viewer().callback_post_draw = &draw;

    Visualize::viewer().callback_key_down = key_down_callback;

    Visualize::viewer().launch_init(true, false, "Meshless Deformations", 0, 0);
    Visualize::viewer().launch_rendering(true);
    simulating = false;
    Visualize::viewer().launch_shut();
}
