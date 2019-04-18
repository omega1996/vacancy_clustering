import pandas as pd
import numpy as np

import mxnet as mx
from bert_embedding import BertEmbedding

from datetime import datetime


co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_13K.pkl')
documents = np.array(co.text).tolist()

print('start: ' + str(datetime.now()))

ctx = mx.gpu(0)
bert_embedding = BertEmbedding(ctx=ctx, model='bert_12_768_12',
                               dataset_name='wiki_cn_cased',
                               max_seq_length=500,
                               batch_size=20)
vectors = bert_embedding(documents)

print('start mean: ' + str(datetime.now()))

new_vectors = []
for vector in vectors:
    new_vectors.append(np.array(vector[1]).mean(0))

print('end: ' + str(datetime.now()))

vectors = np.array(new_vectors)

co['bert_768'] = vectors.tolist()
co = co[['id', 'is_prog', 'is_test', 'label_true', 'title', 'bert_768']]

co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/corpus/df_vacancies_full_ru_13K_bert.pkl')