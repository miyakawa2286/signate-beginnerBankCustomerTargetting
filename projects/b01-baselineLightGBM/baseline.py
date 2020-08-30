import os
print(os.getcwd())
import sys
import pickle

import numpy as np
import lightgbm as lgb

sys.path.append('./my_modules/')
from utils import TARGET, LOCAL_TRAIN_RATIO
from utils import read_dataset
from mls import my_lgb_cross_validation
from utils import make_submission

dpath_to_output = './projects/b01-baselineLightGBM/res'

# define params
lgb_params = {
    'objective': 'binary',
    'is_unbalance': True,
    'metrics': ['auc','binary_logloss'],
    'early_stopping_round': 25,
    'first_metric_only': True,
    'num_iterations': 1600,
    'learning_rate': 0.1,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'device_type': 'cpu',
    }
clf_threshold = 0.5

# load dataset
train,test,sub = read_dataset('./data/')
features = [c for c in train.columns if c!=TARGET]

# cross validation
cv_res = my_lgb_cross_validation(lgb_params,
                                 train,
                                 TARGET,
                                 features,
                                 )
# write cv results
cv_res['eval'].to_csv(os.path.join(dpath_to_output,'cv_eval.csv'))
cv_res['pred'].to_csv(os.path.join(dpath_to_output,'cv_pred.csv'))
cv_res['feature_importance'].to_csv(os.path.join(dpath_to_output,'feature_importance.csv'))
with open(os.path.join(dpath_to_output,'learning_history.pkl'),'wb') as wb:
    pickle.dump(cv_res['learning_history'],wb)
for cv_i,model in cv_res['models'].items():
    with open(os.path.join(dpath_to_output,'cv_models',f'model_{cv_i}.pkl'),'wb') as wb:
        pickle.dump(model,wb)

# make submission
# make local train and val
local_train_size = int(train.shape[0]*LOCAL_TRAIN_RATIO)
local_train_index = np.random.choice(train.index,local_train_size,replace=False)
local_train = train.iloc[train.index.isin(local_train_index)]
local_val = train.iloc[~train.index.isin(local_train_index)]
# make dataset
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
