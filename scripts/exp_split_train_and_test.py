import pandas as pd
import numpy as np

from sklearn.cluster import AffinityPropagation

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_13K_w2v.pkl')
vectors_name = 'w2v_300'

co = data[data.is_prog]

co_train = co[co.is_test == False].sample(4000)
co_test = co[co.is_test]
co_all = pd.concat([co_train, co_test])


# Включена в обучения
X = np.array(co_all[vectors_name])
X = X.tolist()

labels = AffinityPropagation().fit_predict(X)
co_all['label_all'] = labels


# Не включена в обучения
X_train = np.array(co_train[vectors_name])
X_train = X_train.tolist()

X_test = np.array(co_test[vectors_name])
X_test = X_test.tolist()

model = AffinityPropagation()
model.fit(X_train)
labels = model.predict(X_test)
co_test['label_test'] = labels

co_test['label_all'] = co_all.label_all

re = co_test[['id', 'label_test', 'label_all', 'label_true']]

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



print('nunique')
print(co_test.label_test.nunique())

print()
print('adjusted_rand_score')
print(adjusted_rand_score(co_test.label_true, co_test.label_test))

print()
print('homogeneity_completeness_v_measure')
print(homogeneity_completeness_v_measure(co_test.label_true, co_test.label_test))