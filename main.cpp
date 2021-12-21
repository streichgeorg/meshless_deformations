#include <iostream>
#include <thread>
#include <string>

#include <visualization.h>
#include <igl/edges.h>
#include <igl/edge_lengths.h>
#include <igl/readMESH.h>
#include <igl/boundary_facets.h>
#include <Eigen/Sparse>
#include <EigenTypes.h>

#include "init_state.h"
#include "dV_spring_particle_particle_dq.h"
#include "util.h"
#include "cluster.h"
#include "goal_minimization.h"
#include "params.h"

using namespace Eigen;

MatrixXd V;
MatrixXi T;
MatrixXi F;

VectorXd x;
VectorXd xdot;

VectorXd m;

std::vector<Cluster> clusters;

bool simulating = true;

void simulate() {
    int n = x.rows() / 3;

    for(int t = 0; ; t++) {

        VectorXd f_def = VectorXd::Zero(x.size());

        VectorXd N = VectorXd::Zero(x.size() / 3);
        for (auto &cluster : clusters) {
            for (int k : cluster.vertices) N(k) += cluster.weight;
        }

        for (auto &cluster : clusters) {
            VectorXd x_cluster(3 * cluster.vertices.size());
            for (int i = 0; i < cluster.vertices.size(); i++) {
                x_cluster.segment<3>(3 * i) = x.segment<3>(3 * cluster.vertices[i]);
            }

            VectorXd g;
            goal_minimization(g, cluster.Z, cluster.B, cluster.m, x_cluster, use_quadratic);

            for (int i = 0; i < cluster.vertices.size(); i++) {
                VectorXd d = g - x_cluster;
                f_def.segment<3>(3 * cluster.vertices[i]) +=
                    cluster.weight * alpha * d.segment<3>(3 * i) / N(cluster.vertices[i]);
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

        // Add some friction to make the result a bit more realistic looking
        f_ext -= s_friction * xdot;

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

    std::string cluster_type = (argc == 1) ? "tetrahedron" : argv[1];

    if (cluster_type == "single") {
        alpha = 1e-4;
        k_selected = 1e2;
        beta = 0.3;
        s_friction = 5;

        clusters = single_cluster(V);
    } else if (cluster_type == "coarse") {
        alpha = 1e-4;
        k_selected = 1e2;
        beta = 0.3;
        s_friction = 5;

        clusters = clusters_from_file("../data/coarse_bunny.clusters", T, V);
    } else {
        alpha = 1e-1;
        use_quadratic = false;

        clusters = tetrahedron_clusters(T, V);
    }

    m = VectorXd::Constant(V.rows(), M_PI);

    flatten(x, V);
    xdot = VectorXd::Zero(x.size());

    std::thread simulation_thread(simulate);
    simulation_thread.detach();

    //setup libigl viewer and activate
    Visualize::setup(true);
    Visualize::add_object_to_scene(V, F, Vector3d(0.1, 0.5, 0.8));
    Visualize::viewer().callback_post_draw = &draw;

    Visualize::viewer().launch_init(true, false, "Meshless Deformations", 0, 0);
    Visualize::viewer().launch_rendering(true);
    simulating = false;
    Visualize::viewer().launch_shut();
}
