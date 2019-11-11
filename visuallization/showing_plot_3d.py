import re, seaborn as sns, numpy as np, pandas as pd, random
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_clusters_results.csv')

sns.set_style("whitegrid", {'axes.grid': False})

for azim in range(0, 181, 30):
    plt.clf()

    fig = plt.figure(figsize=(24, 24))

    ax = Axes3D(fig, azim=-azim)

    ax.scatter(co.x, co.y, co.z, c=co.label_test, marker='o', s=1)
    # ax.legend()

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/plots/clustering' + str(
        azim) + '.png', format='png', dpi=300)

    plt.show()






plt.clf()

fig = plt.figure(figsize=(24, 24))

ax = Axes3D(fig, azim=-azim)

ax.scatter(d.x, d.y, d.z, c=d.Birch_artm_200, marker='o', s=1)
# ax.legend()

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig('/home/mluser/master8_projects/clustering_vacancies/results/release/plots/result_Birch_artm_3d.png',
    format='png', dpi=300)

plt.show()





t = pd.get_dummies(co.label_true)
t['label_true'] = co.label_true
d = t.groupby(['label_true']).sum()
d = d[[     1,      2,      3,      4,      5,      6,      7,      8,      9,
           10,     11,     12,     13,     14,     15,     16,    101,    102,
          103,    104,    105,    106,    107,    108,    109,    110,    111,
          112,    113,    114,    115,    116,    117,    118,    119,    120,
          121,    122]]
sns.set()
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("hls", 38))
(f, ax) = plt.subplots(1)
d.T.plot(kind='bar', stacked=True, legend=False, ax=ax)
plt.subplots_adjust(bottom=0.1)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Size of clusters')
plt.ylabel('Name of clusters')
plt.savefig(
        '/home/mluser/master8_projects/clustering_vacancies/results/release/plots/counts_labels_colored_n_label_true.svg',
        format='svg', dpi=300)