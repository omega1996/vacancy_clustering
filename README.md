# Experimental Comparison of Unsupervised Approaches in the Task of Separating Specializations within Professions in Job Vacancies

In this article we present an unsupervised approach to analyzing labor market requirements, allowing to solve the problem of discovering latent specializations within broadly defined professions. For instance, for the profession of "programmer", such specializations could be "CNC programmer", "mobile developer", "frontend developer", and so on. We have experimentally evaluated various statistical methods of vector representation of texts: TF-IDF, probabilistic topic modeling, neural language models based on distributional semantics (word2vec, fasttext), and deep contextualized word representation (ELMo and multilingual BERT). We have investigated both pre-trained models, and models trained on the texts of job vacancies in Russian. The experiments were conducted on dataset, provided by online recruitment platforms. We have tested several types of clustering methods: K-means, Affinity Propagation, BIRCH, Agglomerative clustering, and HDBSCAN. In case of predetermined number of clusters (k-means, agglomerative) the best result was achieved by ARTM. However, if the number of clusters was not specified ahead of time, word2vec, trained on our job vacancies dataset, has outperformed other models. The models, trained on our corpora perform much better then pre-trained models with large even multilingual vocabulary.

## Dadasets

430K job vacancies texts
https://drive.google.com/open?id=1Fh6e_AqMgMWwv0w1jyBJOSvP6NF70qNF

22K job vacancies for clustering
https://drive.google.com/open?id=1xw1_F6RLEspQHXzDhmhQYRPg0YxMBbor

2K marked job vacancies
https://drive.google.com/open?id=1T-t6NhTyJ8FtDAR-kNlGzXqKjs8gcSyE
