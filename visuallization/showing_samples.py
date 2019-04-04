import pandas as pd

co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')

co.kmeans_k30.value_counts()

re = co[co.kmeans_k30 == 2].sample(50)[['title', 'class_labels', 'kmeans_k8']]