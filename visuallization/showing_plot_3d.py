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


