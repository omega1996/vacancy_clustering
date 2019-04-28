import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import os
from datetime import datetime
import gensim.downloader as api

model = api.load('word2vec-ruscorpora-300')
model.wv.init_sims()

step = 0

zero = 0

def get_vectorized_avg_w2v_corpus(corpus, model):
    documents = np.array(corpus.lemmatized_text_pos_tags)

    document_vectors = [word_averaging(model, document) for document in documents]
    clean_corpus = corpus
    clean_corpus['w2v'] = pd.Series(document_vectors).values

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
        global zero
        zero = zero + 1
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


d = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')[:10]
d = get_vectorized_avg_w2v_corpus(d, model.wv)
d['w2v_300_rc'] = d.w2v
d = d[['id', 'w2v_300_rc']]
print(zero)
d.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_w2v_300_rc.pkl')