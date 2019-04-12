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






co = pd.read_csv('/home/mluser/master8_projects/clustering_vacancies/results/df_vacancies_full_result_20K_prog_sorted.csv')

corpus = []


t = co[co.aff_d300_euclidean.isin([1153, 1001, 1070])]
t['label_true'] = 3
print(t.shape)
t = t.sample(100)
corpus.append(t)


t = pd.concat(corpus)
corpus = t

data = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_full.pkl')
data.index = data.id

corpus.index = corpus.id
corpus = corpus.sort_values(['label_true', 'aff_d300_euclidean'])
corpus['text'] = data.text
corpus = corpus[['id', 'label_true', 'aff_d300_euclidean', 'title', 'text']]