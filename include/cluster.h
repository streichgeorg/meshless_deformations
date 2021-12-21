#include <Eigen/Dense>
#include <EigenTypes.h>

#include <string>
#include <vector>
#include <set>

// Struct containing information about a cluster as well as some precomputed
// values that are used for the shape matching step
struct Cluster {
    double weight;

    std::vector<int> vertices;

    Eigen::VectorXd m;
    Eigen::MatrixXd Z;
    Eigen::Matrix<double, 9, 9> B;

    Cluster(
        const std::set<int>& vertex_set,
        const Eigen::MatrixXd &V,
        double _weight
    );
};

std::vector<Cluster> clusters_from_file(
    const std::string &fname,
    const Eigen::MatrixXi &T,
    const Eigen::MatrixXd &V
);

std::vector<Cluster> single_cluster(const Eigen::MatrixXd &V);
std::vector<Cluster> tetrahedron_clusters(const Eigen::MatrixXi &T, const Eigen::MatrixXd &V);
