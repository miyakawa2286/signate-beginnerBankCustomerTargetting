import os
print(os.getcwd())
import sys

import numpy as np
import pandas as pd
import category_encoders as ce
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sys.path.append('./my_modules/')
from utils import TARGET, LOCAL_TRAIN_RATIO
from utils import read_dataset

dpath_to_output = './projects/b99-clustering/'

# load dataset
train,test,sub = read_dataset('./data/')
features = [c for c in train.columns if c!=TARGET]
# drop TARGET
train = train.drop(TARGET,axis=1)

# encoding categorical features
# one hot encoding
# train = pd.get_dummies(train,columns=train.columns[train.dtypes=='category'])
# test = pd.get_dummies(test,columns=test.columns[test.dtypes=='category'])
cats = train.columns[train.dtypes=='category'].to_list()
ce_ohe = ce.OneHotEncoder(cols=cats, handle_unknown='value')
train = ce_ohe.fit_transform(train)
test = ce_ohe.transform(test)

# normalize data
train_mean = train.mean()
train_std = train.std()
train = (train-train_mean)/train_std
test = (test-train_mean)/train_std

# # ward clustering
# Z = linkage(train, method='ward', metric="euclidean")
# dendrogram(Z, no_labels=True)
# plt.title('ward')
# plt.savefig(os.path.join(dpath_to_output,'ward_clustering_output.png'))

# k-means clustering
# build k-means
n_clusters = np.arange(4,32,2)
for n_cluster in n_clusters:
    kmeans = KMeans(n_clusters=n_cluster, random_state=2020)
    kmeans.fit(train)
    train['pred'] = kmeans.predict(train)
    test['pred'] = kmeans.predict(test)
    kmeans_output_df = pd.DataFrame(index=np.arange(n_cluster))
    kmeans_output_df.loc[:,'train'] = train.groupby('pred').size()
    kmeans_output_df.loc[:,'test'] = test.groupby('pred').size()
    print('-'*30)
    print(kmeans_output_df)
