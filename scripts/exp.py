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
