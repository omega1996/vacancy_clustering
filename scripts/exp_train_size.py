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

for i in range(10000, 999, -500):
    print('train size ' + str(i))
    co = co.sample(i)
    co_train = co_train.sample(i)
    co_test = co[co.is_test]

    X_train = np.array(co_train[vectors_name])
    X_train = X_train.tolist()

    X_test = np.array(co_test[vectors_name])
    X_test = X_test.tolist()

    model = AffinityPropagation()
    model.fit(X_train)
    labels = model.predict(X_test)
    co_test['label_test'] = labels

    count.append(co_test.label_test.nunique())
    count.append(adjusted_rand_score(co_test.label_true, co_test.label_test))
    count.append(homogeneity_completeness_v_measure(co_test.label_true, co_test.label_test))
