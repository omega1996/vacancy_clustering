from elmo.elmoformanylangs import Embedder
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_42K.pkl')
documents = np.array(co.preprocessed_text)

embedder = Embedder('elmo/model/')

vectors = embedder.sents2elmo(documents)

new_vectors = []
for vector in vectors:
    new_vectors.append(vector.mean(0))

vectors = np.array(new_vectors)

svd = TruncatedSVD(n_components=300).fit(vectors)
vectors = svd.transform(vectors)

co['elmo_300'] = vectors.tolist()
co = co[['id', 'is_prog', 'is_test', 'is_train', 'label_true', 'elmo_300']]

co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_42K_elmo.pkl')