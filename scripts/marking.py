import pandas as pd

co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_result_20K_other.csv')
co.index = co.id

data = pd.read_csv('/home/mluser/master8_projects/pycharm_project_755/data/new/hh_all_corpus.csv', sep='|')
data.index = data.id

co['aff_d300_euclidean'] = co.label
co['link'] = data.alternate_url
co['text'] = data.description
co['title'] = data.name

t = co.drop_duplicates(['text'])
co = t[['id', 'aff_d300_euclidean', 'title', 'link']]

co['freq'] = co.groupby('aff_d300_euclidean')['aff_d300_euclidean'].transform('count')
t = co.sort_values(['freq', 'aff_d300_euclidean'], ascending=False)

co = t
co.to_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_result_20K_other_sorted.csv', index=False)






co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_result_20K_other_sorted.csv')

corpus = []

i = 11

labels = [
        [840],
        [252, 804, 170, 471, 539, 424, 551, 700, 148, 98, 49, 618, 381],
        [14, 245, 801],
        [828, 880, 756, 688, 675, 153, 758],
        [617, 403],
        [528, 62, 739, 413],
        [99, 827, 477, 173, 8],
        [296, 829],
        [157, 533, 812, 634, 873],
        [752, 401],
        [265, 438, 172]
    ]

for l in labels:
    t = co[co.aff_d300_euclidean.isin(l)]
    t['label_true'] = i
    print(t.shape)
    t = t.sample(100)
    corpus.append(t)
    i += 1

t = pd.concat(corpus)
corpus = t

data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full.pkl')
data.index = data.id

corpus.index = corpus.id
corpus = corpus.sort_values(['label_true', 'aff_d300_euclidean'])
corpus['text'] = data.text
corpus = corpus[['id', 'label_true', 'aff_d300_euclidean', 'title', 'text']]