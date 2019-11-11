import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, Birch
from sklearn.cluster import AffinityPropagation

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_w2v.pkl')

X = np.array(co['w2v'])
X = X.tolist()

model = AffinityPropagation()
# model = Birch(n_clusters=None, threshold=0.58)
labels = model.fit_predict(X)

co['labels'] = labels
co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_w2v_result_aff.pkl')

