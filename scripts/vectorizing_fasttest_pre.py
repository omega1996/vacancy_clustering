import pandas as pd
import numpy as np
import gensim

import os
from datetime import datetime

fasttext = gensim.models.FastText.load('/home/mluser/master8_projects/clustering_vacancies/models/187/model.model')
fasttext.wv.init_sims()

step = 0


def get_vectorized_avg_w2v_corpus(corpus, model):
    documents = np.array(corpus.lemmatized_text)

    document_vectors = [word_averaging(model, document) for document in documents]
    clean_corpus = corpus
    clean_corpus['fasttext'] = pd.Series(document_vectors).values

    return clean_corpus


def word_averaging(wv, words):
    all_words, mean = set(), []

    global step
    step = step + 1
    if step % 2000 == 0:
        print('pid_' + str(os.getpid()) + ' ' + str(datetime.now()) + ' count ' + str(step))

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        print('all zero')
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


d = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
d = get_vectorized_avg_w2v_corpus(d, fasttext.wv)
d['fasttext_300_taiga'] = d.fasttext
d = d[['id', 'fasttext_300_taiga']]
d.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_fasttext_300_taiga.pkl')