import os
import sys

import pandas as pd 
import numpy as np

sys.path.append('./my_modules/')
from utils import TARGET
from utils import read_dataset
from mls import my_lgb_cross_validation

train,_,_ = read_dataset('./data/')
features = [c for c in train.columns if c!=TARGET]

# print(sum(ytr==1)/ytr.shape[0])
# print(sum(ytr===0)/ytr.shape[0])

lgb_params = {
    'objective': 'binary',
    'is_unbalance': True,
    'metrics': ['auc','binary_logloss'],
    'early_stopping_round': 25,
    'first_metric_only': True,
    'num_iterations': 800,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'device_type': 'cpu',
    }

res = my_lgb_cross_validation(train,
                              features,
                              TARGET,
                              lgb_params,
                              )

res['cv_res'].to_csv('./projects/b99-crossValidation/cv_res.csv')
res['cv_pred'].to_csv('./projects/b99-crossValidation/cv_pred.csv')

print(1)
