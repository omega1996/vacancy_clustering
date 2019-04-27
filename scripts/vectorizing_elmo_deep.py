from elmo.elmoformanylangs import Embedder
import pandas as pd
import numpy as np
from datetime import datetime

import os
os.environ["TFHUB_CACHE_DIR"] = '/home/mluser/master8_projects/clustering_vacancies/tmp'

from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz")
# elmo([['вопрос', 'жизни', 'Вселенной', 'и', 'вообще', 'всего'], ['42']])

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K.pkl')[:10]
documents = np.array(co.preprocessed_text).tolist()

print(str(datetime.now()))
vectors = elmo(documents)
print(str(datetime.now()))

co['elmo_1024_news'] = np.array(vectors).tolist()
co = co[['id', 'is_prog', 'is_test', 'label_true', 'title', 'elmo_1024_news']]
co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K_elmo_1024_news.pkl')