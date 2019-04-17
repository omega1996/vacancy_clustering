import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

from util import clustering


# for k in range(16, 31, 1):
#     clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans_' + str(k))
#
# for k in range(40, 101, 10):
#     clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans_' + str(k))

k = 16

clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans')
clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1), 'HDBSCAN')
clustering.cluster(AgglomerativeClustering(n_clusters=k), 'Agglomerative')
clustering.cluster(SpectralClustering(n_clusters=k, n_jobs=-1), 'SpectralClustering')
clustering.cluster(SpectralClustering(n_clusters=k, n_jobs=-1, assign_labels='discretize'), 'SpectralClustering')
clustering.cluster(AffinityPropagation(), 'AffinityPropagation')
clustering.cluster(Birch(n_clusters=k), 'Birch')
clustering.cluster_fit_pred(GaussianMixture(n_components=k), 'GaussianMixture')

metrics = {'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
 'canberra': hdbscan.dist_metrics.CanberraDistance,
 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
 'dice': hdbscan.dist_metrics.DiceDistance,
 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
 'hamming': hdbscan.dist_metrics.HammingDistance,
 'haversine': hdbscan.dist_metrics.HaversineDistance,
 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
 'jaccard': hdbscan.dist_metrics.JaccardDistance,
 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
 'l1': hdbscan.dist_metrics.ManhattanDistance,
 'l2': hdbscan.dist_metrics.EuclideanDistance,
 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
 'matching': hdbscan.dist_metrics.MatchingDistance,
 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
 'p': hdbscan.dist_metrics.MinkowskiDistance,
 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}

for m in ['chebyshev', 'euclidean', 'hamming', 'jaccard', 'manhattan']:
    clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1, metric=metrics[m]), 'HDBSCAN')