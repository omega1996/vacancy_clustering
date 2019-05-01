import pandas as pd
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary

from datetime import datetime

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_430K.pkl')
texts = np.array(co.lemmatized_text_pos_tags.apply(eval))
texts = texts.tolist()

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# print(str(datetime.now()))
# lda = gensim.models.LdaModel(corpus, num_topics=500, id2word=dictionary)
# print(str(datetime.now()))
#
# lda.get_document_topics()


print(str(datetime.now()))
lsi = gensim.models.LsiModel(corpus, num_topics=38, id2word=dictionary)
print(str(datetime.now()))
lsi.save('/home/mluser/master8_projects/clustering_vacancies/models/lsi/lsi_38')


co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
co['lsi_38'] = co.lemmatized_text_pos_tags.apply(eval).apply(dictionary.doc2bow).apply(lambda x: np.array(lsi[x])[:,1].tolist())
co = co[['id', 'lsi_38']]
co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_lsi_38.pkl')