import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation

data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v_d2_ru.pkl')

n = 50000
is_run = True
while is_run:
    try:
        print(n)
        co = data[data.is_prog == False].sample(n)

        X = np.array(co['w2v_d2'])
        X = X.tolist()

        from sklearn.metrics.pairwise import cosine_distances
        word_cosine = cosine_distances(X)

        labels = AffinityPropagation(affinity='precomputed').fit_predict(word_cosine)

        co['label'] = labels

        co[['id', 'label']].to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_other.csv', index=False)
        is_run = False
    except:
        n = n - 10000
        is_run = True


# co = data[data.is_prog == False]
#
# X = np.array(co['w2v'])
# X = X.tolist()
#
# labels = AffinityPropagation().fit_predict(X)
#
# co['label'] = labels
#
# co[['id', 'label']].to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_other.csv', index=False)