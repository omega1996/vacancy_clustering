# import hdbscan
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from util import clustering

k = 38

vectors_list = [
    'artm_200',
    'artm_80',
    'bert_768_wmc',
    'elmo_1024_twitter',
    'fast_ai_50',
    'fasttext_300',
    # 'tfidf_300',
    # 'w2v_300',
    # 'artm_500_60',
    # 'elmo_1024_news',
    'tfidf_500',
    'tfidf_80',
    'w2v_tfidf',
    'artm_300_30',
    'artm_300_60',
    'artm_500_30',
]

clustering.cluster(AffinityPropagation(), 'AffinityPropagation', vectors_list)

vectors_list = [
    # 'artm_200',
    # 'artm_80',
    # 'bert_768_wmc',
    # 'elmo_1024_twitter',
    # 'fast_ai_50',
    # 'fasttext_300',
    # 'tfidf_300',
    # 'w2v_300',
    'artm_500_60',
    # 'elmo_1024_news',
    # 'tfidf_500',
    # 'tfidf_80',
    # 'w2v_tfidf',
    'artm_300_30',
    'artm_300_60',
    'artm_500_30',
]

clustering.cluster(AgglomerativeClustering(n_clusters=k), 'Agglomerative', vectors_list)

vectors_list = [
    'artm_200',
    'artm_80',
    'bert_768_wmc',
    'elmo_1024_twitter',
    'fast_ai_50',
    'fasttext_300',
    'tfidf_300',
    'w2v_300',
    'artm_500_60',
    'elmo_1024_news',
    'tfidf_500',
    'tfidf_80',
    'w2v_tfidf',
    'artm_300_30',
    'artm_300_60',
    'artm_500_30',
]

# clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans', vectors_list)

clustering.cluster(SpectralClustering(n_clusters=k, n_jobs=-1), 'SpectralClustering', vectors_list)
clustering.cluster(Birch(n_clusters=None), 'Birch', vectors_list)
clustering.cluster(DBSCAN(n_jobs=-1), 'DBSCAN', vectors_list)

# for k in range(16, 31, 1):
#     clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans_' + str(k))
#
# for k in range(40, 101, 10):
#     clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans_' + str(k))

# clustering.cluster_sparse(KMeans(n_clusters=k, n_jobs=-1), 'KMeans')

# clustering.cluster(SpectralClustering(n_clusters=k, n_jobs=-1, assign_labels='discretize'), 'SpectralClustering')
# clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1, alpha=.5), 'HDBSCAN')
# clustering.cluster_fit_pred(GaussianMixture(n_components=k), 'GaussianMixture')

# for d in range(1, 10, 1):
#     clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1, alpha=d/10), 'HDBSCAN')


# from rake_nltk import Rake
# import numpy as np
#
# t = co[co.label_pred == 33]
# text = np.array(t.text)
# text = ' '.join(text)
#
# r = Rake(language='russian')
# r.extract_keywords_from_text(text)
# top = pd.Series(r.get_ranked_phrases())