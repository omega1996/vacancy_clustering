from sklearn.cluster import AffinityPropagation, Birch
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_w2v.pkl')

X = np.array(co['w2v'])
X = X.tolist()

# model = AffinityPropagation()
# labels = model.fit_predict(X)
#
# co['labels'] = labels

model = KMeans(n_clusters=7, n_jobs=-1)
labels = model.fit_predict(X)
co['labels'] = labels

# data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_split.pkl')
# data.index = data.id
# co.index = co.id
# co['text'] = data.part


t = co[['id', 'text', 'vacancy_id']]
t['l7'] = co.labels
# t = t.sort_values(['labels'])
# tt = t[t.labels == 1]

re = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_w2v_182K_result.pkl')
re.index = re.id
re['l7'] = t.l7
re.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_w2v_182K_result.pkl')



re = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_w2v_182K_result.pkl')
re.index = re.id
re['len'] = re.text.apply(len)

t0 = re[re.l7 == 0]
t2 = re[re.l7 == 2]
t3 = re[re.l7 == 3]
t6 = re[re.l7 == 6]

re = pd.concat([t0, t2, t3, t6])
common = [
    'Условия работы:',
    'Должностные обязанности:',
    'Будет плюсом:',
    'Требования к кандидату:',
    'Плюсом будет:',
    'Основные обязанности:',
    'Личные качества:',
    'Основные требования:',
    'Обязательные требования:',
    'Будет преимуществом:',
    'Требования к кандидатам:',
    'Функциональные обязанности:',
    'Большим плюсом будет:',
    'Ключевые задачи:',
    'Желательные требования:',
]

re = re[~re.text.isin(common)]
re = re[re.len < 150]


n = 3
while n < 100:
    model = KMeans(n_clusters=n, n_jobs=-1)
    labels = model.fit_predict(X)
    re['labels_' + str(n)] = labels
    print(str(n))
    n = n * 2 + 1

re.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_tfudf_300_result.csv')


for i in range(1, 11, 1):
    model = Birch(n_clusters=None, threshold=i/10)
    labels = model.fit_predict(X)
    re['labels_Birch_0' + str(i)] = labels
    co['labels'] = labels
    print('threshold ' + str(i) + ' ' + str(co.labels.nunique()))

re.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_tfudf_300_result.csv')


print('homogeneity_completeness_v_measure: ' + str(homogeneity_completeness_v_measure(re[co.is_test].label_true, co[co.is_test].label_test)))

# homogeneity_completeness_v_measure: (0.4015125147988651, 0.48059427157606205, 0.4375085138420338) aff birch 07
# homogeneity_completeness_v_measure: (0.14698048091837934, 0.26962401590823365, 0.19024983084534666) aff k63
# homogeneity_completeness_v_measure: (0.5821505634559756, 0.5672002076882723, 0.5745781510536495) aff k2763
# homogeneity_completeness_v_measure: (0.8205722822114225, 0.6679414554001922, 0.7364315566489779) birch 07 k2763


re = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_w2v_result.csv')
co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/split/df_vacancies_split_50K_w2v.pkl')
X = np.array(co['w2v'])
X = X.tolist()
silhouette_score(X, re.labels_7)
calinski_harabaz_score(X, re.labels_7)

# si w2v labels_7 0.07450071
# ca w2v labels_7 2643.580897378922

# si w2v label_aff -0.4341435
# ca w2v label_aff 1.002209428538756

for i in [3, 7, 15, 31, 63, 2763]:
    print('ca ' + str(i) + ' ' + str(calinski_harabaz_score(X, re['labels_' + str(i)])))


for i in [5, 6, 7, 8, 9]:
    print('ca ' + str(i) + ' ' + str(calinski_harabaz_score(X, re['labels_Birch_0' + str(i)])))


for i in [3, 7, 15, 31, 63, 2763]:
    print('si ' + str(i) + ' ' + str(silhouette_score(X, re['labels_' + str(i)])))


for i in [5, 6, 7, 8, 9]:
    print('si ' + str(i) + ' ' + str(silhouette_score(X, re['labels_Birch_0' + str(i)])))