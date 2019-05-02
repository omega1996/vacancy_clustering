import pandas as pd
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary

from datetime import datetime

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
texts = np.array(co.lemmatized_text_pos_tags.apply(eval))
texts = texts.tolist()

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(str(datetime.now()))
lda = gensim.models.LdaModel(corpus, num_topics=500, id2word=dictionary)
print(str(datetime.now()))
#
# lda.get_document_topics()


# print(str(datetime.now()))
# lsi = gensim.models.LsiModel(corpus, num_topics=500, id2word=dictionary)
# print(str(datetime.now()))
# lsi.save('/home/mluser/master8_projects/clustering_vacancies/models/lsi/lsi_500_22K')


# co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
# co['lsi_500_22K'] = co.lemmatized_text_pos_tags.apply(eval).apply(dictionary.doc2bow).apply(lambda x: np.array(lsi[x])[:,1].tolist())
# co = co[['id', 'lsi_500_22K']]
# co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_lsi_500_22K.pkl')


def to_vec(x):
    v = np.array(lda[x])
    df = pd.DataFrame(v, columns=['topic', 'prob'])
    df.topic = df.topic.astype('int')
    df.index = df.topic
    re = pd.DataFrame(index=range(0, 500, 1))
    re['prob'] = df.prob
    re.prob = re.prob.fillna(0.0)
    return np.array(re.prob).tolist()


co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
co['lda_500_22K'] = co.lemmatized_text_pos_tags.apply(eval).apply(dictionary.doc2bow).apply(to_vec)
co = co[['id', 'lda_500_22K']]
co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_lda_500_22K.pkl')

