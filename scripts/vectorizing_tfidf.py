import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import pickle

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K.pkl')
documents = np.array(co.lemmatized_text_pos_tags.apply(eval).apply(lambda x: ' '.join(x)))

vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
vectors = vectorizer.fit_transform(documents)

file = open('/home/mluser/master8_projects/clustering_vacancies/data/corpus/vacancies_full_ru_22K_tfidf_all.pkl', 'wb')
pickle.dump(vectors, file)
file.close()

# svd = TruncatedSVD(n_components=300).fit(vectors)
# vectors = svd.transform(vectors)
#
# co['tfidf_300'] = vectors.tolist()
# co = co[['id', 'is_prog', 'is_test', 'label_true', 'title', 'tfidf_300']]
#
# co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K_tfidf_300.pkl')