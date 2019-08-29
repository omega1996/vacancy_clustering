import os
import glob
import artm
import random

import time

import numpy as np

import logging
logging.basicConfig(level = logging.ERROR)#logging.DEBUG)

import matplotlib.pyplot as plt
plt.ioff()

from tqdm import tqdm

import pandas as pd

from files_and_dirs import *

import pickle
def write_to_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

import winsound

# # To configure logging folder
# import artm
lc = artm.messages.ConfigureLoggingArgs()
# lc.log_dir=r'C:\bigartm_log'
# lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)
#
# # To change any other logging parameters at runtime (except logging folder)
lc.minloglevel=3  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
lib.ArtmConfigureLogging(lc)


import pickle
def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        data_new = pickle.load(f)
    return data_new

def write_to_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


class BigARTM():
    save_scores = False
    num_toptokens = 10
    num_document_passes = 1
    num_collection_passes = 30
    base_dir = ""
    
    def __init__(self, name, num_of_topics):
        logging.info('Create BigARTM object. ')
        self.name = name
        self.num_of_topics = num_of_topics
        
        logging.info(f'Param: {num_of_topics}, {BigARTM.num_toptokens}, {BigARTM.num_document_passes}, {BigARTM.num_collection_passes}.')
        self._prepare_folders()

    def _prepare_folders(self):
        dirs = [f'{base_dir}\\matrixes',
                f'{base_dir}\\plots',
                f'{base_dir}\\dictionary',
                f'{base_dir}\\top_tokens',
                f'{base_dir}\\uci',
                f'{base_dir}\\scores'
                f'{base_dir}\\uci\\batches'
                ]

        for dir_ in dirs:
            #print(dir_)
            if not os.path.exists(dir_):
                os.makedirs(dir_)

    def create_batch_vectorizer(self, data_path, data_format, collection_name, batch_folder):
        logging.info(f"Create batch_vectorizer...")
        
        if (len(glob.glob(os.path.join('bow', '.batch'))) < 1):
            self.batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                         data_format=data_format,
                                                         collection_name=collection_name,
                                                         target_folder=batch_folder)
        else:
            self.batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                         data_format=data_format)
        logging.info(f"Create dictionary...")
        self.dictionary = self.batch_vectorizer.dictionary
        logging.info(f"Done")
        
        

    def create_topics_names(self, topics_names=[]):
        if (topics_names == []):
            logging.info('Create topics_names by default.')
            self.topics_names = ['topic_{}'.format(i) for i in range(self.num_of_topics)]
        else:
            logging.info('Create topics_names by user_topics_names.')
            self.topics_names = topics_names

        if (len(self.topics_names) != self.num_of_topics):
            logging.error('Количество тем указанное пользователем не совпадает с количеством тем в модели')
            self.topics_names = []

#    def _create_plsa_model(self):
#        logging.info('Create PLSA model.')
#        self.model_plsa = artm.ARTM(topic_names=self.topics_names,
#                                    cache_theta=True,
#                                    scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=self.dictionary)])
#        self.model_plsa.initialize(dictionary=self.dictionary)

#    def _plsa_add_scores(self):
#        logging.info(f"Add scores to PLSA model.")
#        self.model_plsa.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
#        self.model_plsa.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
#        self.model_plsa.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))
#        self.model_plsa.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=self.num_of_topics))

    def create_artm_model_empty(self):
        logging.info('Create empty ARTM model.')
        self.model_artm = artm.ARTM(topic_names=self.topics_names,
                                    cache_theta=True,
                                    scores=[],
                                    regularizers=[])
        
        self.model_artm.initialize(dictionary=self.dictionary)
        #self.model_artm.show_progress_bars = True

    def save_artm_dictionary(self):
        filename = f"{base_dir}\\dictionary\\{self.name}_dictionary.txt"
        self.dictionary.save_text(filename)

    def create_cooc_dict(self, batches_folder, cooc_file, vocab_file):
        logging.info(f"Create cooc_dict.")        
        self.cooc_dict = self.dictionary
        self.cooc_dict.gather(
                data_path=batches_folder,
                cooc_file_path=cooc_file,
                vocab_file_path=vocab_file,
                symmetric_cooc_values=True)

    def artm_add_scores(self):
        logging.info(f"=== SCORES ===")
        logging.info(f"Add scores to ARTM model.")
        
        logging.info(f"Add score PerplexityScore.")
        self.model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=self.dictionary))
        
        logging.info(f"Add score SparsityPhiScore.")
        self.model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
        
        logging.info(f"Add score SparsityThetaScore.")
        self.model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
        
        
        logging.info(f"Add score TopTokensScore.")
        self.model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=BigARTM.num_toptokens))
        
        logging.info(f"Add score TopicKernelScore.")
        self.model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))

        
        
        logging.info(f"Add score ItemsProcessedScore.")
        self.model_artm.scores.add(artm.TopTokensScore(name='ItemsProcessedScore'))

        logging.info(f"Add score ThetaSnippetScore.")
        self.model_artm.scores.add(artm.TopTokensScore(name='ThetaSnippetScore'))
        
        logging.info(f"Add score TopicMassPhiScore.")
        self.model_artm.scores.add(artm.TopTokensScore(name='TopicMassPhiScore'))

        logging.info(f"Add score ClassPrecisionScore.")
        self.model_artm.scores.add(artm.TopTokensScore(name='ClassPrecisionScore'))
        
        logging.info(f"Add score BackgroundTokensRatioScore.")
        self.model_artm.scores.add(artm.TopTokensScore(name='BackgroundTokensRatioScore'))
        
        #if(self.cooc_dict != None):
        logging.info(f"Add score TopTokensCoherenceScore.")
        self.coherence_score = artm.TopTokensScore(
                name='TopTokensCoherenceScore',
                class_id='@default_class',
                num_tokens=10,
                topic_names=self.topics_names,
                dictionary=self.cooc_dict)    
        self.model_artm.scores.add(self.coherence_score)
    
        
    def add_regularization_artm(self, SparsePhi=0, SparseTheta=0, DecorrelatorPhi=0):
        logging.info(f"=== REGULARIZATION ===")
        logging.info(f"Add regularization to ARTM model.")
        
        self.SparsePhi = SparsePhi
        logging.info(f"Add regularizer SparsePhi. tau = {SparsePhi}")
        self.model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=SparsePhi))
        
        self.SparseTheta = SparseTheta
        logging.info(f"Add regularizer SparseTheta. tau = {SparseTheta}")
        self.model_artm.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=SparseTheta))
        
        self.DecorrelatorPhi = DecorrelatorPhi
        logging.info(f"Add regularizer DecorrelatorPhi. tau = {DecorrelatorPhi}")
        self.model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=DecorrelatorPhi))

#    def print_regularizations(self):
#        print(f"SparsePhi = {self.model_artm.regularizers['SparsePhi'].tau}")
#        print(f"SparseTheta = {self.model_artm.regularizers['SparseTheta'].tau}")
#        print(f"DecorrelatorPhi = {self.model_artm.regularizers['DecorrelatorPhi'].tau}")

    def reup_regularization_artm(self, SparsePhi=0, SparseTheta=0, DecorrelatorPhi=0):
        logging.info(f"reup_regularization")

        self.SparsePhi = SparsePhi
        self.SparseTheta = SparseTheta
        self.DecorrelatorPhi = DecorrelatorPhi

        self.model_artm.regularizers['SparsePhi'].tau = SparsePhi
        self.model_artm.regularizers['SparseTheta'].tau = SparseTheta
        self.model_artm.regularizers['DecorrelatorPhi'].tau = DecorrelatorPhi

    def fit_model(self):
        logging.info(f"Fit models")
        #self.model_plsa.num_document_passes = num_document_passes
        self.model_artm.num_document_passes = BigARTM.num_document_passes
        #self.model_plsa.fit_offline(batch_vectorizer=self.batch_vectorizer,
        #                            num_collection_passes=num_collection_passes)
        self.model_artm.fit_offline(batch_vectorizer=self.batch_vectorizer,
                                    num_collection_passes=BigARTM.num_collection_passes)

    def print_measures_artm(self):
        print('Sparsity Phi: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['SparsityPhiScore'].last_value))
        print('Sparsity Theta: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['SparsityThetaScore'].last_value))
        print('Kernel contrast: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['TopicKernelScore'].last_average_contrast))
        print('Kernel purity: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['TopicKernelScore'].last_average_purity))
        print('Perplexity: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['PerplexityScore'].last_value))

#    def print_measures_plsa_artm(self):
#        print('Sparsity Phi: \t\t{0:.3f} (PLSA) \tvs. \t{1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['SparsityPhiScore'].last_value,
#            self.model_artm.score_tracker['SparsityPhiScore'].last_value))
#        print('Sparsity Theta: \t{0:.3f} (PLSA) \tvs. \t{1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['SparsityThetaScore'].last_value,
#            self.model_artm.score_tracker['SparsityThetaScore'].last_value))
#        print('Kernel contrast: \t{0:.3f} (PLSA) \tvs. \t{1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['TopicKernelScore'].last_average_contrast,
#            self.model_artm.score_tracker['TopicKernelScore'].last_average_contrast))
#        print('Kernel purity: \t\t{0:.3f} (PLSA) \tvs. \t{1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['TopicKernelScore'].last_average_purity,
#            self.model_artm.score_tracker['TopicKernelScore'].last_average_purity))
#        print('Perplexity: \t\t{0:.3f} (PLSA) \tvs. \t{1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['PerplexityScore'].last_value,
#            self.model_artm.score_tracker['PerplexityScore'].last_value))

    def plot_measures_artm(self):
        filename = f"{base_dir}\\plots\\{self.name}_measures.png"

        f = plt.figure()
        ax = f.add_subplot(111)
        f.set_size_inches(10, 7)
        
        ls = random.choice(['-', '--', '-.', ':'])
        marker=random.choice([".",",","o","v","^","<",">","1","2","3",
             "4","8","s","p","P","*","h","H","+","x",
             "X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11])
        
        plt.plot(range(self.model_artm.num_phi_updates),
                 self.model_artm.score_tracker['PerplexityScore'].value, ls=ls, marker=marker, linewidth=2)

        sparsity_phi = 'Sparsity Phi: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['SparsityPhiScore'].last_value)
        sparsity_theta = 'Sparsity Theta: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['SparsityThetaScore'].last_value)
        kernel_contrast = 'Kernel contrast: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['TopicKernelScore'].last_average_contrast)
        kernel_purity = 'Kernel purity: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['TopicKernelScore'].last_average_purity)
        perplexity = 'Perplexity: {0:.3f} (ARTM)'.format(
            self.model_artm.score_tracker['PerplexityScore'].last_value)

        measures = [sparsity_phi, sparsity_theta, kernel_contrast, kernel_purity, perplexity]
        plt.text(0.2, 0.8,'\n'.join(measures), ha='left', va='center', transform=ax.transAxes, fontsize=14)
        #plt.text(150, 0, '\n'.join(measures), fontsize=14)

        plt.title(f'Measures. {self.name}')
        plt.xlabel('Iterations count')
        plt.ylabel('ARTM PerplexityScore.')
        plt.grid(True)
        plt.savefig(filename)
        #plt.show()

#    def plot_measures_plsa_artm(self):
#        filename = f"{base_dir}\\plots\\measures\\{self.name}_measures.png"
#
#        f = plt.figure()
#        ax = f.add_subplot(111)
#        f.set_size_inches(10, 7)
#
#        plt.plot(range(self.model_plsa.num_phi_updates),
#                 self.model_plsa.score_tracker['PerplexityScore'].value, 'b--',
#                 range(self.model_artm.num_phi_updates),
#                 self.model_artm.score_tracker['PerplexityScore'].value, 'r--', linewidth=2)
#
#        sparsity_phi = 'Sparsity Phi: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['SparsityPhiScore'].last_value,
#            self.model_artm.score_tracker['SparsityPhiScore'].last_value)
#        sparsity_theta = 'Sparsity Theta: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['SparsityThetaScore'].last_value,
#            self.model_artm.score_tracker['SparsityThetaScore'].last_value)
#        kernel_contrast = 'Kernel contrast: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['TopicKernelScore'].last_average_contrast,
#            self.model_artm.score_tracker['TopicKernelScore'].last_average_contrast)
#        kernel_purity = 'Kernel purity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['TopicKernelScore'].last_average_purity,
#            self.model_artm.score_tracker['TopicKernelScore'].last_average_purity)
#        perplexity = 'Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
#            self.model_plsa.score_tracker['PerplexityScore'].last_value,
#            self.model_artm.score_tracker['PerplexityScore'].last_value)
#
#        measures = [sparsity_phi, sparsity_theta, kernel_contrast, kernel_purity, perplexity]
#        plt.text(0.2, 0.8,'\n'.join(measures), ha='left', va='center', transform=ax.transAxes, fontsize=14)
#        #plt.text(150, 0, '\n'.join(measures), fontsize=14)
#
#        plt.title(f'Measures. topic={self.num_of_topics}')
#        plt.xlabel('Iterations count')
#        plt.ylabel('PLSA perp. (blue), ARTM perp. (red)')
#        plt.grid(True)
#        plt.savefig(filename)
#        plt.show()

    def plot_score_tracker(self, score):
        f = plt.figure()
        f.set_size_inches(10, 7)
        # plt.plot(range(self.model_plsa.num_phi_updates),
        #          self.model_plsa.score_tracker[score].value, 'b--', linewidth=2)
        ls = random.choice(['-', '--', '-.', ':'])
        marker=random.choice([".",",","o","v","^","<",">","1","2","3",
             "4","8","s","p","P","*","h","H","+","x",
             "X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11])
        
        plt.plot(range(self.model_artm.num_phi_updates),
                 self.model_artm.score_tracker[score].value, marker=marker, ls=ls, linewidth=2)
        plt.title(f'{score}. topics={self.num_of_topics}.', size=20)
        plt.xlabel(f'Iterations count', size=20)
        plt.ylabel(f'{score}',size=20)
        plt.legend(fontsize=20)
        
        plt.grid(True)
        plt.savefig(f"{base_dir}\\plots\\{score}\\{self.name}.png")
        #plt.show()

    def save_matrix_phi(self):
        phi_matrix = self.model_artm.get_phi()
        phi_matrix.head()
        phi_matrix.to_csv(f"{base_dir}\\matrixes\\{self.name}_phi.csv")

    def save_matrix_theta(self):
        theta_matrix = self.model_artm.get_theta()
        theta_matrix.head()
        theta_matrix.to_csv(f"{base_dir}\\matrixes\\{self.name}_theta.csv")

    def save_top_tokens(self, filename=''):
        if(filename==''):
            filename = f"{base_dir}\\top_tokens\\{self.name}"
        
        res_dict_artm = {}
        
        if(len(self.model_artm.topic_names) != len(self.model_artm.score_tracker['TopTokensScore'].last_tokens)):
            print("Присутствуют пустые темы!!!")
        
        for topic_name in self.model_artm.topic_names:
            try:
                value = self.model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name]
            except:
                value = []
            res_dict_artm[topic_name] = value
        write_to_pickle(f"{filename}.pickle", res_dict_artm)

        lst = []
        for topic_name in self.model_artm.topic_names:
            row_lst = []
            row_lst.append(topic_name)
            try:
                row_lst.extend(self.model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])
            except:
                row_lst.extend([])
            lst.append(row_lst)

        df = pd.DataFrame(lst)
        df.to_csv(f"{filename}.csv")
        
    def get_values_by_score_tracker(self, score):
        scores = []
        if score == 'PerplexityScore':
            scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            scores.append((self.name, score, "raw", self.model_artm.score_tracker[score].raw))
            scores.append((self.name, score, "normalizer", self.model_artm.score_tracker[score].normalizer))
            scores.append((self.name, score, "zero_tokens", self.model_artm.score_tracker[score].zero_tokens))
            scores.append((self.name, score, "class_id_info", self.model_artm.score_tracker[score].raw))            
            return scores
        
        if score == 'SparsityPhiScore':
            scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            scores.append((self.name, score, "zero_tokens", self.model_artm.score_tracker[score].zero_tokens))
            scores.append((self.name, score, "total_tokens", self.model_artm.score_tracker[score].total_tokens))
            return scores
        
        if score == 'SparsityThetaScore':
            scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            scores.append((self.name, score, "zero_topics", self.model_artm.score_tracker[score].zero_topics))
            scores.append((self.name, score, "total_topics", self.model_artm.score_tracker[score].total_topics))
            return scores
        
        if score == 'TopTokensScore':
            scores.append((self.name, score, "num_tokens", self.model_artm.score_tracker[score].num_tokens))
            scores.append((self.name, score, "coherence", self.model_artm.score_tracker[score].coherence))
            scores.append((self.name, score, "average_coherence", self.model_artm.score_tracker[score].average_coherence))
            scores.append((self.name, score, "tokens", self.model_artm.score_tracker[score].tokens))
            scores.append((self.name, score, "weights", self.model_artm.score_tracker[score].weights))
            return scores
        
        if score == 'TopicKernelScore':
            scores.append((self.name, score, "tokens", self.model_artm.score_tracker[score].tokens))
            scores.append((self.name, score, "size", self.model_artm.score_tracker[score].size))
            scores.append((self.name, score, "contrast", self.model_artm.score_tracker[score].contrast))
            scores.append((self.name, score, "purity", self.model_artm.score_tracker[score].purity))
            scores.append((self.name, score, "coherence", self.model_artm.score_tracker[score].coherence))
            scores.append((self.name, score, "average_size", self.model_artm.score_tracker[score].average_size))
            scores.append((self.name, score, "average_contrast", self.model_artm.score_tracker[score].average_contrast))
            scores.append((self.name, score, "average_purity", self.model_artm.score_tracker[score].average_purity))
            scores.append((self.name, score, "average_coherence", self.model_artm.score_tracker[score].average_coherence))
            return scores
            
        if score == 'ItemsProcessedScore':
            #scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            return scores

        if score == 'ThetaSnippetScore':
            #scores.append((self.name, score, "document_ids", self.model_artm.score_tracker[score].document_ids))
            #scores.append((self.name, score, "snippet", self.model_artm.score_tracker[score].snippet))
            return scores

        if score == 'TopicMassPhiScore':
            #scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            #scores.append((self.name, score, "topic_mass", self.model_artm.score_tracker[score].topic_mass))
            #scores.append((self.name, score, "topic_ratio", self.model_artm.score_tracker[score].topic_ratio))
            return scores

        if score == 'ClassPrecisionScore':
            #scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            #scores.append((self.name, score, "error", self.model_artm.score_tracker[score].error))
            #scores.append((self.name, score, "total", self.model_artm.score_tracker[score].total))
            return scores
        
        if score == 'BackgroundTokensRatioScore':
            #scores.append((self.name, score, "value", self.model_artm.score_tracker[score].value))
            #scores.append((self.name, score, "tokens", self.model_artm.score_tracker[score].tokens))
            return scores
        
        if score == 'TopTokensCoherenceScore':
            scores.append((self.name, score, "coherence", self.model_artm.score_tracker['TopTokensCoherenceScore'].average_coherence))
            return scores

#%%

def plot_score_trackers_for_few_models(base_dir, score, df_some_score):
    f = plt.figure()
    f.set_size_inches(10, 7)
    
    names_models = []
    ls = random.choice(['-', '--', '-.', ':'])
    markers=[".",",","o","v","^","<",">","1","2","3",
             "4","8","s","p","P","*","h","H","+","x",
             "X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
    for i in range(len(df_some_score)):
        names_models.append(df_some_score.iloc[i]['name_model'])
        plt.plot(range(len(df_some_score.iloc[i]['values'])),
                 df_some_score.iloc[i]['values'], linewidth=2, marker=markers[i], ls=ls)

    plt.title(f'{score[0]}_{score[1]}', size=20)
    plt.xlabel(f'Iterations count', size=20)
    plt.ylabel(f'{score[0]}_{score[1]}', size=20)

    plt.legend(names_models, fontsize=20)
    plt.grid(True)

    names_models_str = '_'.join(names_models)
    print(names_models_str)
    plt.savefig(f"{base_dir}\\plots\\{score[0]}_{score[1]}.png")
    #plt.show()


def create_artm_model(name_model, num_topics, tau_P=0, tau_T=0, dcr=0):
    model = BigARTM(name_model, num_topics)
    model.create_batch_vectorizer(f"{BigARTM.base_dir}\\uci\\", 'bow_uci', 'bow', f"{BigARTM.base_dir}\\batches")
    model.create_topics_names()
    model.create_artm_model_empty()
    model.save_artm_dictionary()
    model.create_cooc_dict(f"{BigARTM.base_dir}\\batches", f"{BigARTM.base_dir}\\cooc\\cooc_tf_", f"{BigARTM.base_dir}\\uci\\vocab.bow.txt")
    model.artm_add_scores()
    model.add_regularization_artm(tau_P, tau_T, dcr)
    model.fit_model()
    print("save")
    model.plot_measures_artm()
    model.save_top_tokens()
    model.save_matrix_phi()
    model.save_matrix_theta()
    return model

def save_model_scores(scores_dir, model):
    print(f"Save scores model: {model.name}")
    scores_tuples = []
    for score in model.model_artm.scores.data.items():
        #print(f"name_score: {score[0]}")
    
        datas = model.get_values_by_score_tracker(score[0])
        for data in datas:
            #print(data[0], data[1], data[2])
            if(data[1] == 'TopicKernelScore' and data[2] == 'tokens'):
                pass
            else:
                scores_tuples.append(data)
    write_to_pickle(f"{dir_}\\scores\\{name_model}.pickle", scores_tuples)
      
#    print(scores_tuples)
#    df = pd.DataFrame(list(scores_tuples), columns = ['name_model', 'score', 'parameter', 'values'])
#    df.head()
#    df.to_csv(f"{scores_dir}\\scores\\{name_model}.csv")
#    df.to_pickle(f"{scores_dir}\\scores\\{name_model}.pickle")

#%%

if __name__ == "__main__":
    
    name_experiment = "vacancies_06_22k"

    #base_dir = "C:\\Users\\Ivan\\PycharmProjects\\all_my_projects_3_6\\bigartm2\\experiments"
    base_dir = "C:\\Users\\Ivan\\PycharmProjects\\all_my_projects_3_6\\bigartm vacancies\\results"

    dir_ = f"{base_dir}\\{name_experiment}"
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    base_dir = dir_
    
    
    BigARTM.num_toptokens = 10
    BigARTM.num_document_passes = 1
    BigARTM.num_collection_passes = 60
    BigARTM.base_dir = base_dir

    models = []
    
    
#    num_topics = 60
#    name_model = f"{num_topics}_{0}_{0}_{0}"
#    model = create_artm_model(name_model, num_topics)
#    models.append(model)
#
#

    num_topics = [80, 300, 500]
    tau_P_list = [0.0, -0.1, -0.25, -0.5]
    tau_T_list = [0.0, -0.1, -0.25, -0.5, -1, -2]
    dcr_list = [0, 1000, 2500, 5000, 10000]
    
    
    param_list = []
    
    for num_topics in num_topics:
        for tau_P in tau_P_list:
            for tau_T in tau_T_list:
                for dcr in dcr_list:
                    tuple_ = num_topics, tau_P, tau_T, dcr
                    param_list.append(tuple_)
    
    #print(param_list)
    #param_list = param_list[::-1]

    # Эксперимент по определению числа тем
    models = []
    
    start_time = time.time()
    
    
    for i in tqdm(range(len(param_list))):        
        param = param_list[i]
        num_topics = param[0]
        tau_P = param[1] 
        tau_T = param[2] 
        dcr = param[3]
        
        lap_time = time.time()
        
        name_model = f"{str(num_topics).zfill(3)}_{tau_P}_{tau_T}_{dcr}_{BigARTM.num_collection_passes}"
        print(i, name_model)
        #print(f"{dir_}\\scores\\{name_model}.pickle")
        
        if(os.path.exists(f"{dir_}\\scores\\{name_model}.pickle")==False):
            model = create_artm_model(name_model, num_topics, tau_P, tau_T, dcr)
            
            
            save_model_scores(f"{dir_}\\scores\\", model)
            #models.append(model)
            
            print(f"--- Lap_time: {(time.time() - lap_time)/60:.2f} Total: {(time.time() - start_time)/60:.2f} ---")
        
#    # Эксперимент с регуляризацией
#    num_topics = 60    
#
#    name_model = f"{num_topics}_{0}_{0}_{0}"
#    model = create_artm_model(name_model, num_topics)
#    models.append(model)
#
#    tau_P = -0.02; tau_T = -0.03; dcr = 2500
#    name_model = f"{num_topics}_{tau_P}_{tau_T}_{dcr}"
#    model = create_artm_model(name_model, num_topics, tau_P, tau_T, dcr)
#    models.append(model)
#    
#    tau_P = -0.02; tau_T = -0.5; dcr = 5000
#    name_model = f"{num_topics}_{tau_P}_{tau_T}_{dcr}"
#    model = create_artm_model(name_model, num_topics, tau_P, tau_T, dcr)
#    models.append(model)
#    
#    tau_P = -0.02; tau_T = -1; dcr = 10000
#    name_model = f"{num_topics}_{tau_P}_{tau_T}_{dcr}"
#    model = create_artm_model(name_model, num_topics, tau_P, tau_T, dcr)
#    models.append(model)
    
    
#    # Эксперимент по определению влияния регуляризации    
#    num_topics = 200; tau_P = 0; tau_T = 0; dcr = 0
#    
#    start_time = time.time()
#    
#    for tau_T in np.arange(0, -1.01, -0.1):
#        lap_time = time.time()
#        name_model = f"{num_topics}_{tau_P}_{tau_T}_{dcr}"
#        model = create_artm_model(name_model, num_topics, tau_P, tau_T, dcr)
#        
#        models.append(model)
#        print(f"--- Lap_time: {time.time() - lap_time} Total: {time.time() - start_time} ---")


#%%
        
#    save_model_scores(f"{dir_}\\scores\\", model)
#    #%%
#
#    print(f"Save scores model: {model.name}")
#    scores_tuples = []
#    for score in model.model_artm.scores.data.items():
#        print(f"name_score: {score[0]}")
#    
#        datas = model.get_values_by_score_tracker(score[0])
#        for data in datas:
#            print(data[0], data[1], data[2])
#            if(data[1] == 'TopicKernelScore' and data[2] == 'tokens'):
#                pass
#            else:
#                scores_tuples.append(data)
#    write_to_pickle(f"{dir_}\\scores\\{name_model}.pickle", scores_tuples)
#
##%%
#    
#    datas = read_from_pickle(f"{dir_}\\scores\\{name_model}.pickle")
#    print(datas)
#%%        
#    print(scores_tuples)
#    df = pd.DataFrame(list(scores_tuples), columns = ['name_model', 'score', 'parameter', 'values'])
#    df.head()
#    df.to_csv(f"{dir_}\\scores\\{name_model}.csv")
#    df.to_pickle(f"{dir_}\\scores\\{name_model}.pickle")
    
#    for d in data:
#        print(d)
#    print('PerplexityScore', )


#%%
#    scores_tuples = []
#    for model in models:
#        print(model.name)
#        for score in model.model_artm.scores.data.items():
#            print(score[0])
#            scores_tuples.extend(model.get_values_by_score_tracker(score[0]))
#            print(len(scores_tuples))
#
#    df = pd.DataFrame(list(scores_tuples), columns = ['name_model', 'score', 'parameter', 'values'])
#    df.to_csv(base_dir+'\\scores.csv')
#    print(df.iloc[0])
#    
##%%
#    # общие графики по всем моделям по каждому score
#    scores = [('PerplexityScore','value'),
#              ('PerplexityScore','raw'),
#              ('PerplexityScore','normalizer'),
#              ('PerplexityScore','zero_tokens'),
#              ('SparsityPhiScore','value'),
#              ('SparsityPhiScore','zero_tokens'),
#              ('SparsityPhiScore','total_tokens'),
#              ('SparsityThetaScore','value'),
#              ('SparsityThetaScore','zero_topics'),
#              ('SparsityThetaScore','total_topics'),
#              #('TopicKernelScore','tokens'),
#              #('TopicKernelScore','size'),
#              #('TopicKernelScore','contrast'),
#              #('TopicKernelScore','purity'),
#              #('TopicKernelScore','coherence'),
#              ('TopicKernelScore','average_size'),
#              ('TopicKernelScore','average_contrast'),
#              ('TopicKernelScore','average_purity'),
#              ('TopicKernelScore','average_coherence'),
#              ('TopTokensCoherenceScore', 'coherence')]
#
##%%
#    
#    #df['score']
##%%
#    for score in scores:
#        df_some_score = df[(df['score']==score[0]) & (df['parameter']==score[1])]
#        print(len(df_some_score))
#        print(score)
#        print(df_some_score.head())
#        plot_score_trackers_for_few_models(base_dir, score, df_some_score)
    
    
#    # звуковой сигнал окончания процесс вычисления
#    duration = 300  # millisecond
#    freq = 440  # Hz
#    winsound.Beep(freq, duration)    
#%%
    #from text_to_speach import *
    import text_to_speach
    text_to_speach.text_to_speach("Задание выполнено!")    
#%%



##
###    for num_topics in range(10, 201, 10):
###        #for tauF in np.arange(-0.01, -0.051, -0.01):
###        #for tauF in [-0.02, -0.04]:
###        for tauF in [-0.02, -0.05]:
###            #for tauT in np.arange(-0.02, -0.1, -0.02):
###            #for tauT in [-0.033, -0.066, -0.099]:
###            for tauT in [-0.03, -0.06]:
###                #for decorelator_phi in np.arange(2500, 2500, 7501):
###                #for decorelator_phi in [2500, 5000, 7500]:
###                for decorelator_phi in [2500, 10000]:
###                    name_model = f"{num_topics}_{tauF}_{tauT}_{decorelator_phi}"
###                    
###                    model = BigARTM(name_model, base_dir, num_topics, num_toptokens, num_document_passes, num_collection_passes)
###                    model.create_batch_vectorizer(f"{data_folder}\\uci\\", 'bow_uci', 'bow', f"{base_dir}\\batches")
###                    model.create_topics_names()
###                    model.create_artm_model_empty()
###                    model.save_artm_dictionary()
###                    model.artm_add_scores()
###                    model.add_regularization_artm(tauF, tauT, decorelator_phi)
###                    model.fit_model()
###                    
###                    model.plot_measures_artm()
###                    model.save_top_tokens()
###                    model.save_matrix_phi()
###                    model.save_matrix_theta()
###                    models.append(model)
#

#%%
print(model.name)
topic_name = 'topic_69'    
value = model.model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name]
print(len(model.model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name]))
print(model.model_artm.score_tracker['TopTokensScore'].last_tokens)
print(value)

#model.save_top_tokens('')
#print('done')