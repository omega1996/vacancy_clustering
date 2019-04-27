import pickle
from sklearn.decomposition import TruncatedSVD
import pandas as pd

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/corpus/vacancies_full_ru_22K_tfidf_all.pkl'
n = 300

file = open(in_file, 'rb')
v = pickle.load(file)
file.close()

svd = TruncatedSVD(n_components=n).fit(v)
x = svd.transform(v)

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K.pkl')
co['tfidf_' + str(n)] = x.tolist()
co = co[['id', 'is_prog', 'is_test', 'label_true', 'title', 'tfidf_' + str(n)]]

co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K_tfidf_' + str(n) + '.pkl')