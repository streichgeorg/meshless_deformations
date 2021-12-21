#include "cluster.h"

#include <iostream>
#include <fstream>

#include "util.h"

using namespace Eigen;

Cluster::Cluster(
    const std::set<int>& vertex_set,
    const MatrixXd &V,
    double _weight
) : weight(_weight) {

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

    std::vector<std::set<int>> cluster_vertices(n_clusters);
    for (int i = 0; i < n_clusters; i++) cluster_vertices[i] = std::set<int>();

    for (int i = 0; i < V.rows(); i++) {
        cluster_vertices[idx_to_cluster[i]].insert(i);
    }

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < n_clusters; j++) {
            cluster_vertices[j] = grow_cluster(T, cluster_vertices[j]);
        }
    }

    std::vector<Cluster> clusters;
    for (int i = 0; i < n_clusters; i++) {
        clusters.emplace_back(cluster_vertices[i], V, 1.0);
    }

    return clusters;
}

std::vector<Cluster> single_cluster(const MatrixXd &V) {
    std::set<int> vertices;
    for (int i = 0; i < V.rows(); i++) vertices.insert(i);
    std::vector<Cluster> clusters = { Cluster(vertices, V, 1.0) };
    return clusters;
}

std::vector<Cluster> tetrahedron_clusters(const MatrixXi &T, const MatrixXd &V) {
    std::vector<Cluster> clusters;
    for (int i = 0; i < T.rows(); i++) {
        std::set<int> vertices;
        for (int j = 0; j < 4; j++) {
            vertices.insert(T(i, j));
        }
        clusters.emplace_back(vertices, V, 1.0);
    }
    return clusters;
}
