import os
print(os.getcwd())
import sys

import numpy as np
import lightgbm as lgb

sys.path.append('./my_modules/')
from utils import TARGET
from utils import read_dataset
from mls import my_lgb_cross_validation
from utils import make_submission

dpath_to_output = './projects/b01-baselineLightGBM/'

# define params
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
clf_threshold = 0.5

# load dataset
train,test,sub = read_dataset('./data/')
features = [c for c in train.columns if c!=TARGET]

# cv
cv_res = my_lgb_cross_validation(train,
                                 features,
                                 TARGET,
                                 lgb_params,
                                 )
cv_res['cv_res'].to_csv(os.path.join(dpath_to_output,'cv_res.csv'))
cv_res['cv_pred'].to_csv(os.path.join(dpath_to_output,'cv_pred.csv'))
print('\n----- cv score -----\n',cv_res['cv_res'].mean(axis=0))

# make dataset
local_train_size = int(train.shape[0]*0.9)
local_train_index = np.random.choice(train.index,local_train_size,replace=False)
local_train = train.iloc[train.index.isin(local_train_index)]
local_val = train.iloc[~train.index.isin(local_train_index)]
lgb_train = lgb.Dataset(local_train[features], local_train[TARGET])
lgb_val = lgb.Dataset(local_val[features], local_val[TARGET])

# train
clf = lgb.train(params=lgb_params,
                train_set=lgb_train,
                valid_sets=[lgb_val],
                )

# predict
predict_proba = clf.predict(test[features], num_iteration=clf.best_iteration)
predict = [0 if i < clf_threshold else 1 for i in predict_proba]

# write predict
make_submission(base_df=sub,
                pred=predict,
                dpath_to_output=dpath_to_output,
                )
