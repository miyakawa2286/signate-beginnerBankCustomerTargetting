import sys

import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import lightgbm as lgb

sys.path.append('./my_modules/')
from utils import LOCAL_TRAIN_RATIO


def get_evaluate(y_test, predict):
    fpr, tpr, _ = metrics.roc_curve(y_test, predict)
    auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)      
    return auc, precision, recall


def my_lgb_cross_validation(params,
                            train,
                            target,
                            features,
                            clf_threshold=0.5,
                            ):
    # build splitter
    skf = StratifiedKFold(n_splits=5,
                          shuffle=True,
                          random_state=2020,
                          )
    # iterate over folds
    eval_df = pd.DataFrame()
    feature_importance_df = pd.DataFrame(index=features)
    learning_history = {}
    models = {}
    for i, (cv_train_index, cv_test_index) in enumerate(skf.split(train[features],train[target])):
        # set seed
        np.random.seed(i)
        # split data
        cv_train = train.iloc[cv_train_index]
        # local train used for training
        # local val used for early stopping
        local_train_size = int(len(cv_train_index)*LOCAL_TRAIN_RATIO)
        local_train_index = np.random.choice(cv_train_index,local_train_size,replace=False)
        local_train = cv_train.iloc[cv_train.index.isin(local_train_index)]
        local_val = cv_train.iloc[~cv_train.index.isin(local_train_index)]
        # cv_test used for evaluation
        cv_test = train.iloc[cv_test_index]
        # make dataset
        lgb_train = lgb.Dataset(local_train[features], local_train[target])
        lgb_val = lgb.Dataset(local_val[features], local_val[target])
        # training
        evals_result = {}
        clf = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_val],
                        valid_names=['val'],
                        evals_result=evals_result,
                        )
        # save training result
        feature_importance_df.loc[:,f'{i+1}th'] = clf.feature_importance(importance_type='gain')
        learning_history[f'{i+1}th'] = evals_result
        models[f'{i+1}th'] = clf
        # prediction
        predict_proba = clf.predict(cv_test[features], num_iteration=clf.best_iteration)
        predict = [0 if i < clf_threshold else 1 for i in predict_proba]
        # save prediction
        train.loc[cv_test_index,'pred'] = predict_proba
        # evaluation
        auc, precision, recall = get_evaluate(cv_test[target].values,predict)
        # save eval result
        eval_df.loc[f'{i+1}th','auc'] = auc
        eval_df.loc[f'{i+1}th','precision'] = precision
        eval_df.loc[f'{i+1}th','recall'] = recall
        eval_df.loc[f'{i+1}th','f'] = (2*precision*recall)/(recall+precision)

    res = {
        'eval': eval_df,
        'pred': train,
        'feature_importance': feature_importance_df,
        'learning_history': learning_history,
        'models': models,
    }
    return res
