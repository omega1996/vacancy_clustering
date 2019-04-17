import pandas as pd
import numpy as np


from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


def cluster(model, cname):
    re = []

    print('=====================================')
    print(cname)

    for name in ['w2v', 'tfidf', 'elmo', 'fasttext']:

        data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_13K_w2v.pkl')
        vectors_name = 'w2v_300'
        co = data[data.is_prog]

        X = np.array(co[vectors_name])
        X = X.tolist()

        labels = model.fit_predict(X)

        co['label_test'] = labels

        print()
        print('-----------------------------------')
        print(name)

        print('nunique')
        print(co[co.is_test].label_test.nunique())

        print()
        print('adjusted_rand_score')
        print(adjusted_rand_score(co[co.is_test].label_true, co[co.is_test].label_test))

        print()
        print('adjusted_mutual_info_score')
        print(adjusted_mutual_info_score(co[co.is_test].label_true, co[co.is_test].label_test))

        print()
        print('homogeneity_completeness_v_measure')
        print(homogeneity_completeness_v_measure(co[co.is_test].label_true, co[co.is_test].label_test))

        hcv = homogeneity_completeness_v_measure(co[co.is_test].label_true, co[co.is_test].label_test)

        a = np.array(
            [cname, name, co[co.is_test].label_test.nunique(),
             adjusted_rand_score(co[co.is_test].label_true, co[co.is_test].label_test),
             adjusted_mutual_info_score(co[co.is_test].label_true, co[co.is_test].label_test),
             hcv[0], hcv[1], hcv[2],
             str(model)])

        re.append(a)

    co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_13K_prog.csv')
    df = pd.DataFrame(re, columns=['name', 'vec', 'n', 'ARI', 'AMI', 'Homogeneity', 'completeness', 'v_measure', 'model'])
    pd.concat([co, df]).to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_13K_prog.csv', index=False)


