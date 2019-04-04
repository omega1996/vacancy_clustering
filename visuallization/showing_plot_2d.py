import pickle
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file = open('/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_split_tfidf.pkl', 'rb')
v = pickle.load(file)
file.close()


t = v[:1000]
svd = TruncatedSVD(n_components=2).fit(t)
x = svd.transform(t)

co = pd.DataFrame(x, columns=['x', 'y'])

plt.clf()
sns.lmplot(data=co, x='x', y='y', hue='kmeans_k8', fit_reg=False, legend=True, legend_out=True, scatter_kws={"s": 100})
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering.png', format='png', dpi=300)