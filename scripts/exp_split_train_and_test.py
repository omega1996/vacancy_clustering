import pandas as pd
import numpy as np

from sklearn.cluster import AffinityPropagation

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v_ru.pkl')
re = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_result_20K_prog_sorted.csv')
tr = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_1K_prog.csv')

data.index = data.id
re.index = re.id
tr.index = tr.id

re['w2v'] = data.w2v
tr['is_marked'] = True
re['is_marked'] = tr.is_marked
re.is_marked = re.is_marked.fillna(False)

co = re
co = co[['id', 'title', 'w2v', 'is_marked']]

co_train = co[co.is_marked == False].sample(4000)
co_test = co[co.is_marked]
co_all = pd.concat([co_train, co_test])


# Включена в обучения
X = np.array(co_all['w2v'])
X = X.tolist()

labels = AffinityPropagation().fit_predict(X)
co_all['label_all'] = labels


# Не включена в обучения
X_train = np.array(co_train['w2v'])
X_train = X_train.tolist()

X_test = np.array(co_test['w2v'])
X_test = X_test.tolist()

model = AffinityPropagation()
model.fit(X_train)
labels = model.predict(X_test)
co_test['label_test'] = labels

co_test['label_all'] = co_all.label_all
co_test['label_true'] = tr.label_true

re = co_test[['id', 'title', 'label_test', 'label_all', 'label_true']]

print('nunique')
print(re.label_test.nunique())
print(re.label_all.nunique())

print()
print('adjusted_rand_score')
print(adjusted_rand_score(re.label_true, re.label_test))
print(adjusted_rand_score(re.label_true, re.label_all))

print()
print('homogeneity_completeness_v_measure')
print(homogeneity_completeness_v_measure(re.label_true, re.label_test))
print(homogeneity_completeness_v_measure(re.label_true, re.label_all))