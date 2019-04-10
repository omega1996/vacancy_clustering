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
