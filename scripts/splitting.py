import pandas as pd
from bs4 import BeautifulSoup

from tqdm import tqdm
tqdm.pandas()

in_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/hh_all_corpus.csv'
out_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_split.csv'


source = pd.read_csv(in_file, sep='|')

co = source[['id', 'name', 'description']]
co['id'] = co.id.astype('str')


def split_text(text):
    result = []
    soup = BeautifulSoup(text, 'html.parser')
    for br in soup.find_all("br"):
        br.replace_with("#BR#")
    for u in soup.find_all(['ul', 'strong']):
        u.unwrap()
    newText = str(soup).replace('#BR#', '</p><p>')
    soup = BeautifulSoup(newText, 'html.parser')
    for t in soup.find_all(['li', 'p'], recursive=True):
        s = t.get_text()
        if s is not None:
            result.append(s)
    return result


co['parts'] = co.description.progress_apply(split_text)

co.index = co.id
t = co['parts'].apply(pd.Series).stack()
t.name = 'part'
t.index = t.index.droplevel(-1)
d = pd.DataFrame(t)
d['vacancy_id'] = d.index

d.part = d.part.str.strip()
d.to_csv(out_file, index=False)
d = pd.read_csv(out_file)
d = d.dropna()
d = d[d.part.str.contains(' ')]
d = d.drop_duplicates(['part', 'vacancy_id'])
d.to_csv(out_file, index=False)