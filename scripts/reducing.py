import pickle
from sklearn.decomposition import TruncatedSVD
import pandas as pd

in_file = '/home/mluser/master8_projects/pycharm_project_755/data/new/vacancies_split_tfidf.pkl'
n = 3

file = open(in_file, 'rb')
v = pickle.load(file)
file.close()

svd = TruncatedSVD(n_components=n).fit(v)
x = svd.transform(v)

co = pd.DataFrame(x, columns=['x', 'y', 'z'])