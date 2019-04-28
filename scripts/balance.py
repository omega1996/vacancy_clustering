import pandas as pd
import numpy as np

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K.pkl')

dfs = []

dfs.append(co[co.is_test & (co.label_true == 1)].sample(150))
dfs.append(co[co.is_test & (co.label_true == 3)].sample(150))

labels = np.array(co.label_true.unique())

for l in labels:
    if (l != 3) & (l != 1):
        test = co[co.is_test & (co.label_true == l)]
        if test.id.count() >= 50:
            dfs.append(test)
        else:
            ad = co[(co.is_test == False) & (co.label_true == l)].sample(50 - test.id.count())
            dfs.append(pd.concat([test, ad]))


df = pd.concat(dfs)
