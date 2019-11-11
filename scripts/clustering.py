import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
column = 'w2v'

co = pd.read_pickle(in_file)
X = np.array(co[column])
X = X.tolist()

labels = KMeans(n_clusters=8, n_jobs=-1).fit_predict(X)
co['kmeans_k8'] = labels

out_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/df_vacancies_split_w2v_result30.pkl'
co[['id', 'kmeans_k8']].to_pickle(out_file)

print('label finish')



result = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')
result['kmeans_k8'] = co['kmeans_k8']
result.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv', index=False)



import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
column = 'w2v'

co = pd.read_pickle(in_file)
X = np.array(co[column])
X = X.tolist()

labels = DBSCAN(n_jobs=-1).fit_predict(X)
result = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')
result['DBSCAN'] = co['DBSCAN']
result.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
column = 'w2v'

co = pd.read_pickle(in_file)
X = np.array(co[column])
X = X.tolist()

labels = AffinityPropagation().fit_predict(X)


import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.countplot(x='label', data=co)
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_AffinityPropagation.png', format='png', dpi=300)



import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
column = 'w2v'

re = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')

data = pd.read_pickle(in_file)
co = data.sample(1000)
X = np.array(co[column])
X = X.tolist()

labels = MeanShift(n_jobs=-1, bandwidth=.6).fit_predict(X)


co['label'] = labels
co.index = co.id
re.index = re.id
co['title'] = re.title
co['class'] = re.class_labels
print(co.label.value_counts())
t = co[co.label == 0][['title', 'label', 'class']]

import seaborn as sns
import matplotlib.pyplot as plt

ba = []
size = []
for i in [.5, .6, .7, .8, .9, 1]:
    labels = MeanShift(n_jobs=-1, bandwidth=i).fit_predict(X)

    co['label'] = labels
    co.index = co.id
    re.index = re.id
    co['title'] = re.title
    co['class'] = re.class_labels
    ba.append(i)
    size.append(co.label.nunique())

t = pd.DataFrame(ba, columns=['width'])
t['size'] = size

plt.clf()
sns.lineplot(x='width', y='size', data=t)
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_MeanShift_width.png', format='png', dpi=300)


import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
sns.lineplot(x=K, y=dis)
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_KMeans_train.png', format='png', dpi=300)


import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
column = 'w2v'

re = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')

data = pd.read_pickle(in_file)
co = data.sample(1000)
X = np.array(co[column])
X = X.tolist()

labels = SpectralClustering(n_jobs=-1, n_clusters=30).fit_predict(X)


import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full_w2v.pkl'
column = 'w2v'

re = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')

data = pd.read_pickle(in_file)
co = data.sample(1000)
X = np.array(co[column])
X = X.tolist()

labels = AgglomerativeClustering(n_clusters=30).fit_predict(X)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

plt.clf()
sns.FacetGrid(d, row='vec', col='name').map(sns.countplot, 'pred')
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels.png', format='png', dpi=300)

name = 'KMeans'
plt.clf()
sns.FacetGrid(d[d.name == name], col='vec').map_dataframe(sns.countplot, 'pred')
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels'+name+'.png', format='png', dpi=300)

name = 'KMeans_artm_200'
plt.clf()
sns.countplot(name, data=co, palette=sns.color_palette("Blues_r"))
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels'+name+'.png', format='png', dpi=300)






names = [
    'KMeans_artm_200',
    'KMeans_tfidf_300',
    'KMeans_w2v_300',
    'Birch_artm_200',
    'Birch_w2v_300',
    'Birch_tfidf_300',
    # 'AffinityPropagation_artm_200',
    # 'AffinityPropagation_tfidf_300',
    # 'AffinityPropagation_w2v_300',
    # 'KMeans_bert_768_wmc',
    # 'Agglomerative_bert_768_wmc',
    # 'AffinityPropagation_bert_768_wmc',
]

a = []

d = co[['id', 'is_prog', 'is_test', 'label_true', 'title', 'text', 'KMeans_artm_200']]
d['pred'] = d.KMeans_artm_200
d['vec'] = 'artm_200'
d['name'] = 'KMeans'
a.append(d[['id', 'is_prog', 'is_test', 'label_true', 'title', 'text', 'pred', 'vec', 'name']])




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

names = [
    'KMeans_artm_200',
    'KMeans_tfidf_300',
    'KMeans_w2v_300',
    'Birch_artm_200',
    'Birch_w2v_300',
    'Birch_tfidf_300',
    # 'AffinityPropagation_artm_200',
    # 'AffinityPropagation_tfidf_300',
    # 'AffinityPropagation_w2v_300',
    # 'KMeans_bert_768_wmc',
    # 'Agglomerative_bert_768_wmc',
    # 'AffinityPropagation_bert_768_wmc',
]

k = 5
co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/release/predict_labels_22K.csv')
co = co[(co.label_true == k) & co.is_test]
df = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/release/predict_labels_2K_trans.csv')
df = df[(df.label_true == k) & df.is_test]
print(str(co.label_true.count()))
print(df.groupby(['name', 'vec']).pred.nunique())
for name in names:
    plt.clf()
    sns.set_style("whitegrid")
    sns.countplot(name, data=co, order=co[name].value_counts().index, palette=sns.color_palette("Blues_r")).set(ylim=(0, 90))
    plt.xlabel(name)
    plt.ylabel('Size of clusters')
    plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels_n_'+ name + '.svg',
        format='svg', dpi=300)






co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/release/predict_labels_22K.csv')
co = co[((co.label_true == 1) | (co.label_true == 3)) & co.is_test]
for name in names:
    print(name)
    print('adjusted_rand_score: ' + str(adjusted_rand_score(co.label_true, co[name])))
    print('adjusted_mutual_info_score: ' + str(adjusted_mutual_info_score(co.label_true, co[name])))
    print('homogeneity_completeness_v_measure: ' + str(homogeneity_completeness_v_measure(co.label_true, co[name])))
    print()


co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/release/predict_labels_22K.csv')
name = 'Birch_tfidf_300'
for i in co.label_true.unique():
    t = co[(co.label_true != i) & co.is_test]
    print(name + str(i))
    print('adjusted_rand_score: ' + str(adjusted_rand_score(t.label_true, t[name])))
    print('adjusted_mutual_info_score: ' + str(adjusted_mutual_info_score(t.label_true, t[name])))
    print('homogeneity_completeness_v_measure: ' + str(homogeneity_completeness_v_measure(t.label_true, t[name])))
    print()




k = 26
co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/release/predict_labels_22K.csv')
co = co[(co.Birch_w2v_300 == k) & co.is_test]
# df = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/release/predict_labels_2K_trans.csv')
# df = df[(df.Birch_w2v_300 == k) & df.is_test]
print(str(co.label_true.count()))
# print(df.groupby(['name', 'vec']).pred.nunique())
for name in names:
    plt.clf()
    sns.countplot(name, data=co, palette=sns.color_palette("Blues_r"))
    plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels'+str(k) + name + '.png',
        format='png', dpi=300)



names = [
    'artm_80',
    'artm_200',
    'artm_300_30',
    'artm_500_30'
]

size = [80, 200, 300, 500]
ami_KMeans = [0.6558, 0.6875, 0.6869, 0.6938]
ami_AffinityPropagation = [0.3326, 0.2998, 0.2992, 0.2869]
ami_Agglomerative = [0.646, 0.6709, 0.6597, 0.647]
ami_Birch = [0.6219, 0.6682, 0.5079, 0.5106]

plt.clf()
sns.lineplot(x='size', y='AMI', hue='clustering', data=df).set_title('ARTM')

plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/size_ami.png',
        format='png', dpi=300)

ari = [
0.3906,
0.4531,
0.4387,
0.4376,
0.1187,
0.1003,
0.0995,
0.0939,
0.3832,
0.4465,
0.414,
0.382,
0.4153,
0.4812,
0.2158,
0.2328
]


v = [
0.6905,
0.7189,
0.7172,
0.7284,
0.6478,
0.6539,
0.6527,
0.6529,
0.686,
0.7051,
0.6988,
0.689,
0.6811,
0.7251,
0.6259,
0.6311
]


t = pd.get_dummies(co.label_true)
t['Birch_artm_200'] = co.Birch_artm_200
d = t.groupby(['Birch_artm_200']).sum()
d = d[[     1,      2,      3,      4,      5,      6,      7,      8,      9,
           10,     11,     12,     13,     14,     15,     16,    101,    102,
          103,    104,    105,    106,    107,    108,    109,    110,    111,
          112,    113,    114,    115,    116,    117,    118,    119,    120,
          121,    122]]
sns.set()
sns.set_palette(sns.color_palette("hls", 38))
d.T.plot(kind='bar', stacked=True, legend=False)
plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels_colored_Birch_artm_200.png',
        format='png', dpi=300)





labels = ['lda_500_22K', 'lsi_500_22K', 'tfidf_300', 'fasttext_300_taiga',
       'elmo_1024_news', 'elmo_1024_twitter', 'elmo_300_wiki',
       'elmo_1024_wiki', 'bert_768_wmc', 'fasttext_300', 'artm_200',
       'w2v_300_tfidf', 'w2v_300']

# labels = ['fasttext_300_taiga', 'lda_500_22K', 'fasttext_300', 'artm_200',
#        'tfidf_300', 'w2v_300']

co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_22K.csv')
vecs = ['artm_200', 'bert_768_wmc', 'elmo_1024_news',
       'elmo_1024_twitter', 'fasttext_300', 'tfidf_300',
       'w2v_300', 'w2v_tfidf', 'fasttext_300_taiga', 'elmo_1024_wiki',
       'elmo_300_wiki', 'lsi_500_22K', 'lda_500_22K', 'w2v_2']
co = co[co.vec.isin(vecs)]
co = co[co.name == 'AffinityPropagation']
co = co[co.vec != 'w2v_2']
df = pd.DataFrame(co.groupby(['vec'])['n'].max())
df['Number of clusters'] = df['n']
df['Model'] = df.index
df = df[['Model', 'Number of clusters']]
df = df.sort_values(['Number of clusters'], ascending=False)

plt.clf()
sns.set_style("whitegrid")
(f, ax) = plt.subplots(1)
sns.barplot(x='Model', y='Number of clusters', data=df, ax=ax, palette=sns.color_palette("Blues_r")).set_xticklabels(labels, rotation=90)
plt.subplots_adjust(bottom=0.35)
plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/count_by_methods_n_AffinityPropagation.svg',
        format='svg', dpi=300)








t = pd.get_dummies(co.label_true)
t['label_true'] = co.label_true
d = t.groupby(['label_true']).sum()
d = d[[     1,      2,      3,      4,      5,      6,      7,      8,      9,
           10,     11,     12,     13,     14,     15,     16,    101,    102,
          103,    104,    105,    106,    107,    108,    109,    110,    111,
          112,    113,    114,    115,    116,    117,    118,    119,    120,
          121,    122]]
sns.set()
sns.set_palette(sns.color_palette("hls", 38))
d.columns = [
'1: 1C programmers, CRM specialists without web orientation',
'2: 1C Bitrix programmers, CMS specialists',
'3: PHP programmers, web developers',
'4: Backend developers with sql, pl/sql, db knowledge',
'5: Android, iOS, mobile developers',
'6: Java desktop programmers',
'7: C# .NET ASP.NET developers with web orientation',
'8: Frontend, JavaScript developers',
'9: C# desktop developers without web orientation',
'10: C++, Qt programmers',
'11: gGame developers, level designers',
'12: Site makers, http specialists',
'13: Automation technology specialists',
'14: CNC-operator, machine programmer',
'15: Microcontroller programmers, engineer programmers, IOT, embedded system developers',
'16: Data scientists, data analytics on python, R',
'101: Shop assistant',
'102: System Administrator',
'103: Technician installer engineer',
'104: Sales Manager',
'105: Project Managers',
'106: Business Analysts',
'107: Marketing, promotion, content managers',
'108: Tech support',
'109: Testers and QA',
'110: SAP Consultant',
'111: Designers',
'112: DevOps',
'113: Directors, heads of departments, branches',
'114: SEO-specialists, Search Engine Specialists',
'115: Heads, leaders and directors of 1C',
'116: Regional Sales Managers',
'117: Engineers, Design engineers, constructors',
'118: Consultant Analyst 1C',
'119: Information Security',
'120: Online Store / Internet Project Manager',
'121: Technical Support Engineer',
'122: Network engineers, network administrators',
]
(f, ax) = plt.subplots(1)
d.T.plot(kind='bar', stacked=True, legend=False, ax=ax)
plt.subplots_adjust(bottom=0.5)
plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels_colored_n_label_true.png',
        format='png', dpi=300)