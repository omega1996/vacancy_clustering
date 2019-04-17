import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results_ru_13K_prog.csv')
co = co[co.vec == 'w2v']

plt.clf()
sns.lineplot(x=co.n, y=co.Homogeneity, label='Homogeneity')
sns.lineplot(x=co.n, y=co.completeness, label='completeness')
sns.lineplot(x=co.n, y=co.v_measure, label='v_measure')
sns.lineplot(x=co.n, y=co.ARI, label='ARI')
sns.lineplot(x=co.n, y=co.AMI, label='AMI')
plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering_KMeans_k_metrics.png', format='png', dpi=300)