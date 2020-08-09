import sys

import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import lightgbm as lgb


def get_evaluate(y_test, predict):
    fpr, tpr, thr_arr = metrics.roc_curve(y_test, predict)
    auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)      
    return auc, precision, recall


def my_lgb_cross_validation(train,
                            features,
                            target,
                            params,
                            clf_threshold=0.5,
                            ):

    skf = StratifiedKFold(n_splits=5,
                          shuffle=True,
                          random_state=2020,
                          )
    
    cv_res = pd.DataFrame()
    for i, (cv_train_index, cv_test_index) in enumerate(skf.split(train[features],train[target])):
        
        local_train_size = int(len(cv_train_index)*0.9)
        local_train_index = np.random.choice(cv_train_index,local_train_size,replace=False)

        # split data 
        cv_train = train.iloc[cv_train_index]
        local_train = cv_train.iloc[cv_train.index.isin(local_train_index)]
        local_val = cv_train.iloc[~cv_train.index.isin(local_train_index)]
        cv_test = train.iloc[cv_test_index]
        
        # print('-'*30)
        # print(f'cv_train: {cv_train.shape}')
        # print(f' ratio of class==1: {sum(cv_train.y==1)/cv_train.shape[0]}')
        # print(f'local_train: {local_train.shape}')
        # print(f' ratio of class==1: {sum(local_train.y==1)/local_train.shape[0]}')
        # print(f'local_val: {local_val.shape}')
        # print(f' ratio of class==1: {sum(local_val.y==1)/local_val.shape[0]}')
        # print(f'cv_test: {cv_test.shape}')
        # print(f' ratio of class==1: {sum(cv_test.y==1)/cv_test.shape[0]}')

        # make dataset
        lgb_train = lgb.Dataset(local_train[features], local_train[target])
        lgb_val = lgb.Dataset(local_val[features], local_val[target])

        # train
        clf = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_val],
                        )

        # predict
        predict_proba = clf.predict(cv_test[features], num_iteration=clf.best_iteration)
        predict = [0 if i < clf_threshold else 1 for i in predict_proba]

        # evaluation
        auc, precision, recall = get_evaluate(cv_test[target].values,predict)
        #print(f'\nauc: {auc}\nprecision: {precision}\nrecall: {recall}')

        # save eval result
        cv_res.loc[f'{i+1}th','auc'] = auc
        cv_res.loc[f'{i+1}th','precision'] = precision
        cv_res.loc[f'{i+1}th','recall'] = recall
        cv_res.loc[f'{i+1}th','f'] = (2*precision*recall)/(recall+precision)

        # save prediction
        train.loc[cv_test_index,'pred'] = predict_proba

    res = {
        'cv_res': cv_res,
        'cv_pred': train,
    }
    return res
