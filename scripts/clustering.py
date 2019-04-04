import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
out_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_split_w2v_result30.pkl'
column = 'w2v'

co = pd.read_pickle(in_file)
X = np.array(co[column])
X = X.tolist()

labels = KMeans(n_clusters=8, n_jobs=-1).fit_predict(X)
co['kmeans_k8'] = labels

co[['id', 'kmeans_k8']].to_pickle(out_file)

print('label finish')



result = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')
result['kmeans_k8'] = co['kmeans_k8']
result.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv', index=False)
