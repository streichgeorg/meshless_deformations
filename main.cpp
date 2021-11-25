#include <iostream>
#include <thread>

#include <visualization.h>
#include <igl/edges.h>
#include <igl/edge_lengths.h>
#include <igl/readMESH.h>
#include <igl/boundary_facets.h>
#include <Eigen/Dense>
#include <EigenTypes.h>

#include <init_state.h>

using namespace Eigen;

void goal_minimization(

) {

}

VectorXd q;
VectorXd qdot;

MatrixXd V; //vertices of simulation mesh
MatrixXi T; //faces of simulation mesh
MatrixXi F; //faces of simulation mesh

bool simulating = true;

void simulate() {
    while(simulating) {
        std::cout << "simulating" << std::endl;
        q *= 1 + 1e-5;
    }
}

bool draw(igl::opengl::glfw::Viewer & viewer) {

    Visualize::update_vertex_positions(0, q);

    return false;
}

int main(int argc, char **argv) {

    std::cout<<"Start Our Project\n";

    //load geometric data
    igl::readMESH("../data/coarse_bunny.mesh", V, T, F);
    igl::boundary_facets(T, F);
    F = F.rowwise().reverse().eval();

    init_state(q, qdot, V);

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
