import pandas as pd
import numpy as np
from flair.data import Sentence

from flair.embeddings import BertEmbeddings, DocumentMeanEmbeddings, ELMoEmbeddings

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_22K.pkl')
texts = np.array(co.text.sample(10).apply(Sentence))

word_embedding = BertEmbeddings('bert-base-multilingual-cased')
document_embeddings = DocumentMeanEmbeddings([word_embedding])
document_embeddings.embed(texts.tolist())
a = np.array(sentence.get_embedding().tolist())