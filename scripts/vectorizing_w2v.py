import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import os
from datetime import datetime

word2vec = Word2Vec.load('/home/mluser/master8_projects/clustering_vacancies/models/w2v/w2v_model_on_text_pos_tags')
word2vec.wv.init_sims()

step = 0


def get_vectorized_avg_w2v_corpus(corpus, model):
    documents = np.array(corpus.lemmatized_text_pos_tags.apply(eval))

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
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


d = pd.read_csv('/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_full.csv')
d = get_vectorized_avg_w2v_corpus(d, word2vec.wv)
d = d[['id', 'w2v']]
d.to_pickle('/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_full_w2v.pkl')