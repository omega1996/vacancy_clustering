# from bert_serving.server.helper import get_args_parser
# from bert_serving.server import BertServer
# args = get_args_parser().parse_args(['-model_dir', '/home/mluser/master8_projects/clustering_vacancies/models/bert/rubert_cased_L-12_H-768_A-12_v1/',
#                                      '-port', '5555',
#                                      '-port_out', '5556',
#                                      '-max_seq_len', 'NONE',
#                                      '-pooling_strategy', 'NONE'])
# server = BertServer(args)
# server.start()


import pandas as pd
import numpy as np

from bert_serving.client import BertClient
bert_client = BertClient(check_length=False)

co = pd.read_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K.pkl')
documents = np.array(co.text).tolist()

vectors = bert_client.encode(documents)

# vectors = np.array([np.array([np.array(r) for r in [[e for e in lst if e] for lst in v] if len(r) > 0]).mean(0) for v in vectors])

result = []
count = 0
for v in vectors:
    inner = []
    for i in v:
        b = False
        for n in i:
            if n:
                b = True
        if b:
            inner.append(i)

    if len(inner) == 512:
        count += 1
    result.append(np.array(inner).mean(0))

result = np.array(result)

vectors = result

co['bert_768_ru'] = vectors.tolist()
co = co[['id', 'bert_768_ru']]

co.to_pickle('/home/mluser/master8_projects/clustering_vacancies/data/release/df_vacancies_full_ru_22K_bert_768_ru.pkl')