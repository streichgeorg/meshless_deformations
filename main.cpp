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
using namespace std;

double dt = 1e-4;
double alpha = 0.001;

double k_selected = 1e7;

MatrixXd V;
MatrixXi T;
MatrixXi F;

VectorXd x;
VectorXd xdot;

VectorXd Xbar;

VectorXd m;
Vector3d C;

MatrixXd clu;
VectorXd app;

void polar_rotation(Matrix3d& R, const Matrix3d& A) {
    SelfAdjointEigenSolver<Matrix3d> es(A.transpose() * A);

    Matrix3d D = es.eigenvalues().asDiagonal();
    Matrix3d P = es.eigenvectors();

    Matrix3d S = P * D.cwiseSqrt() * P.transpose();
    R = A * S.inverse();
}

void center_of_mass(Vector3d& r, const VectorXd& x, const VectorXd& m) {
    int n = x.size() / 3;

    r.setZero();
    for (int i = 0; i < n; i++) r += m(i) * x.segment<3>(3 * i);
    r /= m.sum();
}



void clusters(
    MatrixXd &clu, VectorXd &app, 
    const VectorXd& Xbar, const VectorXd& m, const double N
) {

    cout << "Computing clusters...";

    cout << endl;
    size_t n = Xbar.rows() / 3;

    clu.resize(n, N);
    clu.setConstant(-1);

    app.resize(n, 1);
    app.setConstant(0);

  /*  cout << "before" << endl;

    clu.resize(1, n);
    for (size_t j = 0; j < n; j++) {
        clu(0, j) = j;
    }
    app.setConstant(1);

    cout << clu << endl;
    cout << app.transpose() << endl;

    return;*/

    for (size_t i = 0; i < n; i++) {

        for (size_t j = 0; j < N; j++) {

            int closest = -1;
            bool alreadyThere = true;
            while (alreadyThere) {
                alreadyThere = false;
                closest++;
                for (size_t jj = 0; jj <j ; jj++) {
                    if (clu(i,jj) == closest) {
                        alreadyThere = true;
                        break;
                    }
                }
            }
           
            Vector3d closestXbar = Xbar.block<3, 1>(3*closest, 0);

            for (size_t k = 1; k < n; k++) {
                if ((Xbar.block<3, 1>(3 * i, 0) - Xbar.block<3, 1>(3 * k, 0)).norm() <
                    (Xbar.block<3, 1>(3 * i, 0) - closestXbar).norm()) {
                    bool alreadyInList = false;
                    for (size_t x = 0; x < N; x++) {
                        if (clu(i, x) == k) {
                            alreadyInList = true;
                            break;
                        }
                    }
                    if (!alreadyInList) {
                        closestXbar = Xbar.block<3, 1>(3 * k, 0);
                        closest = k;                      
                    }
                }
            }
            clu(i, j) = closest;
            app(closest) += 1;
        }
    }
    cout << " done." << endl;
   // cout << clu << endl;
   // cout << app << endl;
}


// LINEAR GOAL MINIZATION
//void goal_minimization(
//    Matrix3d &R, Vector3d &c,
//    const VectorXd &Xbar, const VectorXd &x, const VectorXd &m
//) {
//
//    center_of_mass(c, x, m);
//
//    Matrix3d Arot = Matrix3d::Zero();
//    Matrix3d Asym = Matrix3d::Zero();
//
//    int n = Xbar.rows() / 3;
//
//    for (int i = 0; i < n; i++) {
//        Arot += m(i) * (x.segment<3>(3 * i) - c) * Xbar.segment<3>(3 * i).transpose();
//        Asym += m(i) * (Xbar.segment<3>(3 * i) * Xbar.segment<3>(3 * i).transpose());
//    }
//
//
//    Matrix3d A = Arot * Asym.inverse();
//    A /= pow(A.determinant(), 1. / 3);
//
//    Matrix3d RR;
//    polar_rotation(RR, Arot);
//    
//    double beta = 0.8;
//
//    R = beta * A + (1 - beta) * RR; 
//}


void goal_minimization_2(
    Matrix3d &R, Vector3d &c,
    const VectorXd &X, const VectorXd &x, const VectorXd &m, const Vector3d &C
) {
    center_of_mass(c, x, m);

    Matrix3d Arot = Matrix3d::Zero();

    int n = X.rows() / 3;

    for (int i = 0; i < n; i++) {
        Arot += m(i) * (x.segment<3>(3 * i) - c) * (X.segment<3>(3 * i) - C).transpose();
    }

    polar_rotation(R, Arot);
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

        
    int Nc = clu.rows(); // # of clusters
    int N = clu.cols();  // # of vertices in a cluster
    
    cout << "Nc:" << Nc << " N:" << N << endl;

    cout << "center of mass: " << C.transpose() << endl;

    for(int t = 0; ; t++) {
        // set deformation force to zero 
        VectorXd f_def = VectorXd(3 * n, 1).setConstant(0);

        // for every cluster i
        for (size_t i = 0; i < Nc; i++) {
            VectorXd X_cluster, m_cluster, x_cluster;
            x_cluster.resize(3 * N);
            X_cluster.resize(3 * N);
            m_cluster.resize(3 * N);

            // for all the vertices j contained in the cluster i
            for (size_t j = 0; j < N; j++) {
                X_cluster.block<3, 1>(3 * j, 0) = Xbar.block<3, 1>(3 * clu(i, j), 0) + C;
                x_cluster.block<3, 1>(3 * j, 0) = x.block<3, 1>(3 * clu(i, j), 0);
                m_cluster(j) = m(clu(i,j)) / app(clu(i,j));
            }

            
            // get center of mass of the cluster in the undeformed space C_cluster
            Vector3d C_cluster;
            center_of_mass(C_cluster, X_cluster, m_cluster);

            // solve the minimization problem
            Matrix3d R_cluster;
            Vector3d c_cluster;
            goal_minimization_2(R_cluster, c_cluster, X_cluster, x_cluster, m_cluster, C_cluster);

            // compute the goal positions of the vertices contained in the cluster
            MatrixXd matrix_X_cluster = Map<MatrixXd>(X_cluster.data(), 3, N);
            matrix_X_cluster.colwise() -= C_cluster; // compute "Xbar" for cluster
            MatrixXd G = R_cluster * matrix_X_cluster;
            G.colwise() += c_cluster;
            VectorXd g = Map<VectorXd>(matrix_X_cluster.data(), 3 * N);

            // add deformation forces to every vertex j contained in the cluster i
            for (size_t j = 0; j < N; j++) {
               f_def.block<3, 1>(3 * clu(i, j), 0) += 1 / app(clu(i, j)) * alpha * (g.block<3, 1>(3 * j, 0) - x_cluster.block<3, 1>(3 * j, 0));
            }
        }
        
        //cout << "finished all clusters" << endl;
        // $$ WHY IS m.size() DIFFERENT FROM XBAR SIZE /3 ? 

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

    center_of_mass(C, Xbar, m);

    flatten(Xbar, V.rowwise() - C.transpose());

    flatten(x, V);
    xdot = VectorXd::Zero(x.size());

    clusters(clu, app, Xbar, m, 8);


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
