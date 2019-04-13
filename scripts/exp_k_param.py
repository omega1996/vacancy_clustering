import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_42K_w2v.pkl')
vectors_name = 'w2v_300'

co = data[data.is_prog]

co_train = co[co.is_train].sample(4000)
co_test = co[co.is_test]

X_train = np.array(co_train[vectors_name])
X_train = X_train.tolist()

X_test = np.array(co_test[vectors_name])
X_test = X_test.tolist()

count = []
ars = []
hcv = []

for i in range(10, 41, 1):
    print('k = ' + str(i))
    model = KMeans(n_clusters=i, n_jobs=-1)
    model.fit(X_train)
    labels = model.predict(X_test)
    co_test['label_test'] = labels

    count.append(co_test.label_test.nunique())
    ars.append(adjusted_rand_score(co_test.label_true, co_test.label_test))
    hcv.append(homogeneity_completeness_v_measure(co_test.label_true, co_test.label_test))


hcv = np.array(hcv)
count = np.array(count)
ars = np.array(ars)

df = pd.DataFrame(hcv)
df['count'] = count
df['ars'] = ars

df.columns = ['homogeneity', 'completeness', 'v_measure', 'count', 'ars']
df['k'] = range(10, 41, 1)

co = df[['k', 'count', 'ars', 'homogeneity', 'completeness', 'v_measure']]
co.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_prog_k.csv', index=False)


import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.lineplot(x=co.k, y=co['count'])
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_KMeans_k_count.png', format='png', dpi=300)

sns.lineplot(x=co.k, y=co.homogeneity, label='homogeneity')
sns.lineplot(x=co.k, y=co.completeness, label='completeness')
sns.lineplot(x=co.k, y=co.v_measure, label='v_measure')
sns.lineplot(x=co.k, y=co.ars, label='ars')
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_KMeans_k_metrics.png', format='png', dpi=300)