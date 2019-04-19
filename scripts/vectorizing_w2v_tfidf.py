import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import os
from datetime import datetime

from util.vectorizer import TfidfEmbeddingVectorizer


co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K.pkl')

X_tfidf = np.array(co.lemmatized_text_pos_tags.apply(eval).apply(lambda x: ' '.join(x)))
X_w2v = np.array(co.lemmatized_text_pos_tags.apply(eval))

model = Word2Vec.load('/home/mluser/master8_projects/clustering_vacancies/models/w2v/w2v_model_on_text_pos_tags')
model.wv.init_sims()

w2v = dict(zip(model.wv.index2word, model.wv.syn0))

vectors = TfidfEmbeddingVectorizer(w2v).fit(X_tfidf, None).transform(X_w2v)
co['w2v_tfidf'] = vectors.tolist()
co = co[['id', 'is_prog', 'is_test', 'label_true', 'title', 'w2v_tfidf']]
co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K_w2v_tfidf.pkl')

