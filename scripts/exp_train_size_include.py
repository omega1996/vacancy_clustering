import pandas as pd
import numpy as np

from sklearn.cluster import AffinityPropagation

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_42K_w2v.pkl')
vectors_name = 'w2v_300'

co = data[data.is_prog]
co_train = co[co.is_train]

count = []
ars = []
hcv = []

for i in range(20000, 999, -500):
    print('train size ' + str(i))
    co = co.sample(i)
    co_train = co_train.sample(i)
    co_test = co[co.is_test]
    co_all = pd.concat([co_train, co_test])

    X_all = np.array(co_all[vectors_name])
    X_all = X_all.tolist()

    model = AffinityPropagation()
    labels = model.fit_predict(X_all)
    co_all['label_test'] = labels
    co_test = co_all[co_all.is_test]

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
df['size'] = range(20000, 999, -500)

co = df[['size', 'count', 'ars', 'homogeneity', 'completeness', 'v_measure']]
co.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_prog_size_in.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.lineplot(x=co['size'], y=co['count'])
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_Aff_size_count_prog_in.png', format='png', dpi=300)

plt.clf()
sns.lineplot(x=co['size'], y=co.homogeneity, label='homogeneity')
sns.lineplot(x=co['size'], y=co.completeness, label='completeness')
sns.lineplot(x=co['size'], y=co.v_measure, label='v_measure')
sns.lineplot(x=co['size'], y=co.ars, label='ars')
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_Aff_size_metrics_prog_in.png', format='png', dpi=300)