#include "cluster.h"

#include <iostream>
#include <fstream>

#include "util.h"

using namespace Eigen;

Cluster::Cluster(
    const std::set<int>& vertex_set,
    const MatrixXd &V,
    double weight_
) : weight(weight_) {

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

    // Precompute the basis expansion for the vertices in undeformed space
    Z = MatrixXd(9, n);
    Z << Xbarr,
         Xbarr.pow(2),
         Xbarr.row(0) * Xbarr.row(1), Xbarr.row(1) * Xbarr.row(2), Xbarr.row(0) * Xbarr.row(2);

    // Precompute the linear, quadratic transorm for the vertices
    Matrix<double, 9, 9> B_inv = Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < n; i++) {
        B_inv += m(i) * Z.col(i) * Z.col(i).transpose();
    }
    B = B_inv.inverse();
}

std::set<int> grow_cluster(const MatrixXi &T, const std::set<int> &S) {
    std::set<int> result;
    for (int i = 0; i < T.rows(); i++) {
        bool contained = false;
        for (int j = 0; j < 4; j++) {
            if (S.count(T(i, j)) > 0) {
                contained = true;
                break;
            }
        }

        if (contained) {
            for (int j = 0; j < 4; j++) result.insert(T(i, j));
        }
    }

    return result;
}

std::vector<Cluster> clusters_from_file(
    const std::string &fname,
    const MatrixXi &T,
    const MatrixXd &V
) {
    std::ifstream f(fname);

    if (!f.is_open()) {
        std::cout << "Failed to read clusters file" << std::endl;
        exit(-1);
    }

    std::vector<int> idx_to_cluster;

    int n_clusters = 0;
    int cluster;
    while (f >> cluster) {
        idx_to_cluster.push_back(cluster);
        n_clusters = std::max(n_clusters, cluster + 1);
    }

    f.close();

    if (!idx_to_cluster.size() == V.rows()) {
        std::cout << "Clusters has wrong format" << std::endl;
    }

    n_clusters += 1;

    std::vector<std::set<int>> cluster_vertices(n_clusters, std::set<int>());

    for (int i = 0; i < V.rows(); i++) {
        cluster_vertices[0].insert(i);
        cluster_vertices[idx_to_cluster[i] + 1].insert(i);
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < n_clusters; j++) {
            cluster_vertices[i] = grow_cluster(T, cluster_vertices[i]);
        }
    }

    std::vector<Cluster> clusters;
    for (int i = 0; i < n_clusters; i++) {
        std::cout << cluster_vertices[i].size() << std::endl;
        clusters.emplace_back(cluster_vertices[i], V, (i > 0) ? 1.0 : 0.0);
    }

    return clusters;
}


