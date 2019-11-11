import pandas as pd
import numpy as np
import gensim

in_file = '/home/mluser/master8_projects/clustering_vacancies/data/df_vacancies_split.pkl'
out_file = '/home/mluser/master8_projects/clustering_vacancies/models/w2v_300_split_13M/w2v_model_on_text_pos_tags'

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

d = pd.read_pickle(in_file)
a = np.array(d.lemmatized_text_pos_tags.apply(eval))

w2v_model = gensim.models.Word2Vec(a, min_count=2, iter=100, size=300, sg=0, workers=32)
w2v_model.save(out_file)