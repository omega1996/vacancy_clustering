from elmo.elmoformanylangs import Embedder
import pandas as pd
import numpy as np
from datetime import datetime

import os
os.environ["TFHUB_CACHE_DIR"] = '/home/mluser/master8_projects/clustering_vacancies/tmp'

from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from deeppavlov.models.embedders.abstract_embedder import ABCMeta

elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz")
# elmo([['вопрос', 'жизни', 'Вселенной', 'и', 'вообще', 'всего'], ['42']])

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
documents = np.array(co.preprocessed_text).tolist()

print(str(datetime.now()))
vectors = elmo(documents)
print(str(datetime.now()))

co['elmo_1024_wiki'] = np.array(vectors).tolist()
co = co[['id', 'elmo_1024_wiki']]
co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_elmo_1024_wiki.pkl')