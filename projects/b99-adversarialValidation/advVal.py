import os
print(os.getcwd())
import sys

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sys.path.append('./my_modules/')
from utils import TARGET, LOCAL_TRAIN_RATIO
from utils import read_dataset
from mls import my_lgb_cross_validation

dpath_to_output = './projects/b99-adversarialValidation/'

# load dataset
train,test,sub = read_dataset('./data/')
features = [c for c in train.columns if c!=TARGET]

# make adv data
adv_target = 'adv_label'
# 0 for train, 1 for test data
train[adv_target] = 0
test[adv_target] = 1
# put dummy value on TARGET
test[TARGET] = 0
adv_data = pd.concat([train,test],axis=0)
# dtype changed by concat
# trans string to categgory
adv_data['job'] = adv_data['job'].astype('category')

# get adverarial prediction
# make dataset
lgb_train = lgb.Dataset(adv_data[features], adv_data[adv_target])
# training
clf = lgb.train(params={'objective': 'binary',
                        'metrics': ['auc'],
                        'num_iterations': 250,
                        'learning_rate': 0.1,
                        'device_type': 'cpu',
                        },
                train_set=lgb_train,
                )
# predict train or test
adv_data['pred'] = clf.predict(adv_data[features])

# cross validation with adv training data
# iterate over adv_clf_thresholds
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
adv_clf_thersholds = np.arange(0,0.55,0.05)
scores = []
for adv_clf_thershold in adv_clf_thersholds:
    print(adv_clf_thershold)
    # get sub train data, which has hign similarity with test data
    adv_train = adv_data.loc[(adv_data[adv_target]==0)&(adv_data['pred']>adv_clf_thershold)].reset_index()
    # cross validation
    cv_res = my_lgb_cross_validation(lgb_params,
                                     adv_train,
                                     TARGET,
                                     features,
                                     )
    # save cv evals
    scores.append(cv_res['eval']['auc'].mean())

fig,axs = plt.subplots(figsize=(10,10))
axs.plot(adv_clf_thersholds,scores)
fig.savefig(os.path.join(dpath_to_output,'adv_thresholds_vs_scores.png'))
