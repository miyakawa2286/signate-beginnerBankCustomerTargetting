import os
import sys

from sklearn.model_selection import StratifiedKFold

sys.path.append('./my_modules/')
from utils import read_dataset

TARGET = 'y'

train,_,_ = read_dataset('./data/')
Xtr = train[[c for c in train.columns if c!=TARGET]]
ytr = train[TARGET]

# print(sum(ytr==1)/ytr.shape[0])
# print(sum(ytr==0)/ytr.shape[0])

skf = StratifiedKFold(n_splits=5,
                      shuffle=True,
                      random_state=2020,
                      )
for cv_train_index, cv_test_index in skf.split(Xtr, ytr):
    #print("cv train:", cv_train_index, "cv TEST:", cv_test_index)
    
    cv_Xtr = Xtr.iloc[cv_train_index]
    cv_ytr = ytr[cv_train_index]
    cv_Xte = Xtr.iloc[cv_test_index]
    cv_yte = ytr[cv_test_index]
    
    # print(sum(ytr[cv_train_index]==1)/ytr[cv_train_index].shape[0])
    # print(sum(ytr[cv_train_index]==0)/ytr[cv_train_index].shape[0])

print(1)
