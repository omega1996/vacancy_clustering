import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_13K_tfidf.pkl')
vectors_name = 'tfidf_300'
co = data[data.is_prog]

co_train = co[co.is_test == False]
co_test = co[co.is_test]

count = []
ari = []
ami = []
hcv = []

for i in range(12000, 999, -500):
    print('train size ' + str(i))
    co_train = co_train.sample(i)

    co_all = pd.concat([co_train, co_test])

    X_all = np.array(co_all[vectors_name])
    X_all = X_all.tolist()

    labels = KMeans(n_clusters=16, n_jobs=-1).fit_predict(X_all)
    co_all['label_test'] = labels
    co_test = co_all[co_all.is_test]

    count.append(co_test.label_test.nunique())
    ari.append(adjusted_rand_score(co_test.label_true, co_test.label_test))
    ami.append(adjusted_mutual_info_score(co_test.label_true, co_test.label_test))
    hcv.append(homogeneity_completeness_v_measure(co_test.label_true, co_test.label_test))


hcv = np.array(hcv)
count = np.array(count)
ari = np.array(ari)
ami = np.array(ami)

df = pd.DataFrame(hcv)
df['count'] = count
df['ari'] = ari
df['ami'] = ari

df.columns = ['homogeneity', 'completeness', 'v_measure', 'count', 'ari', 'ami']
df['size'] = range(12000, 999, -500)

co = df[['size', 'count', 'ari', 'ami', 'homogeneity', 'completeness', 'v_measure']]
co.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_prog_size_kmeans_tfidf.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.lineplot(x=co['size'], y=co['count'])
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_Aff_size_count_prog_kmeans_tfidf.png', format='png', dpi=300)

plt.clf()
sns.lineplot(x=co['size'], y=co.homogeneity, label='homogeneity')
sns.lineplot(x=co['size'], y=co.completeness, label='completeness')
sns.lineplot(x=co['size'], y=co.v_measure, label='v_measure')
sns.lineplot(x=co['size'], y=co.ari, label='ari')
sns.lineplot(x=co['size'], y=co.ami, label='ami')
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_Aff_size_metrics_prog_kmeans_tfidf.png', format='png', dpi=300)