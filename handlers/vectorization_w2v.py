from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec
import os


word2vec = Word2Vec.load(os.path.join(os.getcwd(), "../big_word2vec/big_word2vec_model_CBOW"))
word2vec.wv.init_sims()





def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])


def get_vectorized_avg_w2v_corpus(corpus, model):
    documents = corpus['processed_text'].tolist()

    document_vectors = [word_averaging(model, document) for document in documents]
    clean_corpus = corpus
    clean_corpus['vectors'] = pd.Series(document_vectors).values

    return clean_corpus


def most_similar(infer_vector, vectorized_corpus, topn=10):
    df_sim = vectorized_corpus
    df_sim['similarity'] = df_sim['vectors'].apply(
        lambda v: cosine_similarity([infer_vector], [v.tolist()])[0, 0])
    df_sim = df_sim.sort_values(by='similarity', ascending=False).head(n=topn)
    return df_sim
