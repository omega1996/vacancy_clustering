import pandas as pd
from util.preprocessing import process_text
import swifter

in_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_split.csv'
out_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_split_temp.csv'

d = pd.read_csv(in_file)

d['lemmatized_text_pos_tags'] = d.part.swifter\
    .progress_bar(enable=True)\
    .set_npartitions(npartitions=None)\
    .set_dask_threshold(dask_threshold=20)\
    .allow_dask_on_strings(enable=True)\
    .apply(lambda text: process_text(text)['lemmatized_text_pos_tags'])

d.to_csv(out_file, index=False)