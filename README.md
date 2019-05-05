# Experimental Comparison of Unsupervised Approaches in the Task of Separating Specializations within Professions in Job Vacancies

Abstract. In this article we present an unsupervised approach to ana- lyzing labor market requirements, allowing to solve the problem of dis- covering latent specializations within broadly defined professions. For instance, for the profession of ”programmer”, such specializations could be ”CNC programmer”, ”mobile developer”, ”frontend developer”, and so on. We have experimentally evaluated various statistical methods of vector representation of texts: TF-IDF, probabilistic topic modeling, neu- ral language models based on distributional semantics (word2vec, fast- text), and deep contextualized word representation (ELMo and multilin- gual BERT). We have investigated both pre-trained models, and mod- els trained on the texts of job vacancies in Russian. The experiments were conducted on dataset, provided by online recruitment platforms. We have tested several types of clustering methods: K-means, Affinity Propagation, BIRCH, Agglomerative clustering, and HDBSCAN. In case of predetermined number of clusters (k-means, agglomerative) the best result was achieved by ARTM. However, if the number of clusters was not specified ahead of time, word2vec, trained on our job vacancies dataset, has outperformed other models. The models, trained on our corpora per- form much better then pre-trained models with large even multilingual vocabulary.
