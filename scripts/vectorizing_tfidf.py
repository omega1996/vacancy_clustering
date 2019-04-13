import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_42K.pkl')
documents = np.array(co.lemmatized_text_pos_tags.apply(eval).apply(lambda x: ' '.join(x)))

vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
vectors = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=300).fit(vectors)
vectors = svd.transform(vectors)

co['tfidf_300'] = vectors.tolist()
co = co[['id', 'is_prog', 'is_test', 'is_train', 'label_true', 'tfidf_300']]

co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_42K_tfidf.pkl')