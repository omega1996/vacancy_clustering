from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
# from sklearn.cluster import Birch
from sklearn.cluster import KMeans

from util import clustering


clustering.cluster(KMeans(n_clusters=20, n_jobs=-1), 'KMeans')

# labels = hdbscan.HDBSCAN(core_dist_n_jobs=-1).fit_predict(X)
# labels = AgglomerativeClustering(n_clusters=20).fit_predict(X)
# labels = AffinityPropagation().fit_predict(X)
# labels = SpectralClustering(n_clusters=20, n_jobs=-1).fit_predict(X)
# model = GaussianMixture(n_components=20).fit(X)
# labels = model.predict(X)
# labels = Birch(n_clusters=20).fit_predict(X)
# labels = KMeans(n_clusters=20, n_jobs=-1).fit_predict(X)