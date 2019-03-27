# -*- coding: utf-8 -*-
import pandas as pd

import codecs
import os
import pymorphy2
from string import ascii_lowercase, digits, whitespace

os.getcwd()
morph = pymorphy2.MorphAnalyzer()
cyrillic = u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
allowed_characters = ascii_lowercase + digits + cyrillic + whitespace


def complex_preprocess(text, additional_allowed_characters = "+#"):
    return ''.join([character if character in set(allowed_characters+additional_allowed_characters) else ' ' for character in text.lower()]).split()


def lemmatize(tokens, filter_pos):
    '''Produce normal forms for russion words using pymorphy2
    '''
    lemmas = []
    tagged_lemmas = []
    for token in tokens:
        parsed_token = morph.parse(token)[0]
        norm = parsed_token.normal_form
        pos = parsed_token.tag.POS
        if pos is not None:
            if pos not in filter_pos:
                lemmas.append(norm)
                tagged_lemmas.append(norm + "_" + pos)
        else:
            lemmas.append(token)
            tagged_lemmas.append(token+"_")

    return lemmas, tagged_lemmas


def process_text(full_text, filter_pos=("PREP", "NPRO", "CONJ")):
    '''Process a single text and return a processed version
    '''
    single_line_text = full_text.replace('\n',' ')
    preprocessed_text = complex_preprocess(single_line_text)
    lemmatized_text, lemmatized_text_pos_tags = lemmatize(preprocessed_text, filter_pos=filter_pos)

    return {"full_text" : full_text,
            "single_line_text": single_line_text,
            "preprocessed_text": preprocessed_text,
            "lemmatized_text": lemmatized_text,
            "lemmatized_text_pos_tags": lemmatized_text_pos_tags}


def process_text_documents(text_files_directory, filter_pos=("PREP", "NPRO", "CONJ")):
    for file in os.listdir(text_files_directory):
        if os.path.isfile(os.path.join(text_files_directory, file)):
            with codecs.open(os.path.join(text_files_directory, file), encoding='utf-8') as f:
                full_text = f.read()
                doc_dict = process_text(full_text)
                doc_dict["filename"] = file
                yield doc_dict



# def similarity(competences, zyn, topn=5):
#     df_result = pd.DataFrame(columns=['similarity', 'full_text', 'full_text_match', 'zyn_text',
#                                       'id_discipline', 'type', 'zyn_index'],
#                              index=None)
#     match_index = 0
#     for index, sample in competences.iterrows():
#         similar_docs = most_similar(sample['vectors'], zyn, topn=topn)[['full_text', 'similarity',
#                                                                         'zyn_text', 'id_discipline',
#                                                                         'type', 'zyn_index']]
#         similar_docs['full_text_match'] = sample['full_text_match']
#         df_result = pd.concat([df_result, similar_docs], ignore_index=True)
#         match_index += 1
#         print(index)
#         print(match_index)
#     return df_result
#
#
# def matching_parts(competence, zyn, part, topn=5):
#
#     zyn['full_text'] = zyn[part]
#     zyn['processed_text'] = zyn['full_text'].apply(lambda text: process_text(str(text))['lemmatized_text_pos_tags'])
#     zyn = get_vectorized_avg_w2v_corpus(zyn, word2vec.wv)
#
#     competences = pd.DataFrame(columns=['full_text_match'])
#     competences['full_text_match'] = pd.Series(competence)
#     # лемматизируем
#     competences['processed_text'] = competences['full_text_match'].apply(lambda text: process_text(str(text))['lemmatized_text_pos_tags'])
#     competences = get_vectorized_avg_w2v_corpus(competences, word2vec.wv)
#     return similarity(competences, zyn, topn=topn)


