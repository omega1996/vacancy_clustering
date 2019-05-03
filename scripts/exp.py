import hdbscan
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd

from util import clustering

k = 38

vectors_list = [
    # 'artm_200',
    # 'artm_80',
    'bert_768_wmc',
    # 'elmo_1024_twitter',
    # 'fast_ai_50',
    # 'fasttext_300',
    # 'tfidf_300',
    # 'w2v_300',
    # 'artm_500_60',
    # 'elmo_1024_news',
    # 'tfidf_500',
    # 'tfidf_80',
    # 'w2v_tfidf',
    # 'artm_300_30',
    # 'artm_300_60',
    # 'artm_500_30',
    # 'lsi_500',
    # 'lsi_300',
    # 'fasttext_300_taiga',
    # 'elmo_1024_wiki',
    # 'w2v_2',
    # 'elmo_300_wiki',
    # 'elmo_200_wiki',
    # 'elmo_500_wiki',
    # 'elmo_100_wiki',
    # 'elmo_50_wiki',
    # 'elmo_300_wiki',
    # 'w2v_38',
    # 'lsi_500_22K',
    # 'lda_500_22K',
    # 'w2v_500',
    # 'w2v_1000',
    # 'fast_ai_300',
]

clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans', vectors_list)
clustering.cluster(AgglomerativeClustering(n_clusters=k), 'Agglomerative', vectors_list)


# clustering.cluster(Birch(n_clusters=None, threshold=0.48), 'Birch', ['w2v_38'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.1), 'Birch', ['w2v_2'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.3), 'Birch', ['lda_500_22K'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.2), 'Birch', ['fasttext_300_taiga'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.58), 'Birch', ['w2v_300'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.2), 'Birch', ['tfidf_80'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.3), 'Birch', ['tfidf_300'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.3), 'Birch', ['tfidf_500'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.11), 'Birch', ['artm_80'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.13), 'Birch', ['artm_200'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.11), 'Birch', ['artm_300_30'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.13), 'Birch', ['artm_300_60'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.11), 'Birch', ['artm_500_30'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.12), 'Birch', ['artm_500_60'])
# clustering.cluster(Birch(n_clusters=None, threshold=0.2), 'Birch', ['fasttext_300'])

clustering.cluster(AffinityPropagation(), 'AffinityPropagation', vectors_list)
#
# for name in vectors_list:
#     clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1), 'HDBSCAN', name)






# a = []
# for i in range(1, 11, 1):
#     a.append(clustering.cluster(Birch(n_clusters=None, threshold=i/10), 'Birch' + str(i), vectors_list))
#     # a.append(clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1, alpha=i/10), 'HDBSCAN' + str(i/10), vectors_list))
#
# t = pd.concat(a).sort_values(['vec', 'AMI', 'v_measure'], ascending=False)

# clustering.cluster(DBSCAN(n_jobs=-1), 'DBSCAN', vectors_list)
# clustering.cluster(SpectralClustering(n_clusters=k, n_jobs=-1), 'SpectralClustering', vectors_list)

# for k in range(16, 31, 1):
#     clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans_' + str(k))
#
# for k in range(40, 101, 10):
#     clustering.cluster(KMeans(n_clusters=k, n_jobs=-1), 'KMeans_' + str(k))

# clustering.cluster_sparse(KMeans(n_clusters=k, n_jobs=-1), 'KMeans')

# clustering.cluster(SpectralClustering(n_clusters=k, n_jobs=-1, assign_labels='discretize'), 'SpectralClustering')

# a = []
# met = [
# 'braycurtis',
#  'canberra',
#  'chebyshev',
#  'cityblock',
#  'dice',
#  'euclidean',
#  'hamming',
#  'infinity',
#  'jaccard',
#  'kulsinski',
#  'l1',
#  'l2',
#  'manhattan',
#  'p',
#  'russellrao',
#  'sokalsneath',
# ]
#
# for m in met:
#     a.append(clustering.cluster(hdbscan.HDBSCAN(core_dist_n_jobs=-1, metric=m), 'HDBSCAN_' + m, ['w2v_38']))
#
# t = pd.concat(a).sort_values(['vec', 'AMI', 'v_measure'], ascending=False)

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