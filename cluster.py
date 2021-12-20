import igl
import meshplot as mp
from sklearn.cluster import spectral_clustering

mp.offline()

sv, sf = igl.read_triangle_mesh('data/coarse_bunny.mesh')
A = igl.adjacency_matrix(sf)
labels = spectral_clustering(A)

with open('data/coarse_bunny_clusters', 'w') as f:
    for l in labels:
        f.write(str(l))
        f.write('\n')

mp.plot(sv, sf, c=labels)
