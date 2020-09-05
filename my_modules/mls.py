import os
import sys
import pickle
import glob
import time

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree, linear_model, neighbors, ensemble, neural_network
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

sys.path.append('./my_modules/')
from utils import LOCAL_TRAIN_RATIO
from utils import plot_confusion_matrix


def get_evaluate(y_test, predict):
    fpr, tpr, _ = metrics.roc_curve(y_test, predict)
    auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)      
    return auc, precision, recall


def my_lgb_cross_validation(
    params,
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
        # split data into cv train and test
        cv_train = train.iloc[cv_train_index]
        cv_test = train.iloc[cv_test_index]
        # split cv train into local train and val
        # local train used for training
        # local val used for early stopping
        local_train_size = int(len(cv_train_index)*LOCAL_TRAIN_RATIO)
        local_train_index = np.random.choice(cv_train_index,local_train_size,replace=False)
        local_train = cv_train.iloc[cv_train.index.isin(local_train_index)]
        local_val = cv_train.iloc[~cv_train.index.isin(local_train_index)]
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


def agg_cv_models(train,
                  test,
                  features,
                  target,
                  dpath_to_models,
                  ):
    # iterate over cv models
    cv_model_names = []
    for fpath_to_model in glob.glob(os.path.join(dpath_to_models,'*')):
        print(fpath_to_model)
        # load model
        with open(fpath_to_model,'rb') as rb:
            model = pickle.load(rb)
            cv_model_name = fpath_to_model.split('/')[-1].split('.')[0]
            cv_model_names.append(cv_model_name)
        # get train prediction
        train[cv_model_name] = model.predict(train[features], num_iteration=model.best_iteration)
        # get test prediction
        test[cv_model_name] = model.predict(test[features], num_iteration=model.best_iteration)
    
    # fit meta model
    meta_model = LogisticRegression(class_weight='balanced',random_state=2020)
    meta_model.fit(train[cv_model_names],train[target])
    # get agg prediction
    agg_pred = meta_model.predict(test[cv_model_names])

    res = {
        'agg_pred': agg_pred,
        'cv_model_pred_train': train[cv_model_names],
        'cv_model_pred_test': test[cv_model_names],
    }
    return res


def get_oofp(
    X_train_stm_1: pd.DataFrame,
    X_test_stm_1: pd.DataFrame,
    target,
    models: list,
    subsampling_num: int = 5,
    subsampling_ratio: float = 0.3,
    ):
    '''
    get out-of-fold prediction.
    stage m-1 -> stage m.
    
    Refs
    ------
    [1] strategy, https://www.kaggle.com/general/18793
    
    '''
    # init
    X_train_stm = pd.DataFrame(index=X_train_stm_1.index) 
    X_test_stm = pd.DataFrame(index=X_test_stm_1.index)
    # iterate over subsampling_num
    cv_res = pd.DataFrame()
    total_elapsed_time = 0
    for i in range(subsampling_num):
        # define cv
        skf = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=i,
            )
        # feature sampling
        sub_train = X_train_stm_1.copy()
        sub_train = sub_train.sample(
            frac=subsampling_ratio, 
            axis=1, 
            random_state=i,
            )
        # iterate over models
        for name,model in models.items():
            name = name+'_'+'sub'+str(i+1)
            # use cv predict
            # out-of-folds prediction [1]
            start = time.time()
            cv_pred = cross_val_predict(
                estimator=model,
                X=sub_train.values,
                y=target.values,
                cv=skf,
                method='predict_proba',
                )[:,1] # output prob(Survived=1|X)
            elapsed_time =  time.time() - start
            total_elapsed_time += elapsed_time
            # get scores
            score = roc_auc_score(y_true=target.values,y_score=cv_pred)
            #score = accuracy_score(y_true=target.values, y_pred=np.int16(cv_pred>0.5))
            print('Running Time: {:0=4}[sec], score: {:0=4} Model: {}'.format(round(elapsed_time,1),round(score,3),name))
            X_train_stm.loc[:,name] = cv_pred
            cv_res.loc[name,'auc'] = score
            # fit
            model.fit(sub_train.values,target.values)
            # predict test
            X_test_stm.loc[:,name] = model.predict_proba(X_test_stm_1.loc[:,sub_train.columns].values)[:,1]
    
    return X_train_stm, X_test_stm, cv_res, total_elapsed_time


def my_stacking(
    train: pd.DataFrame,
    test: pd.DataFrame,
    models: list,
    ):
    '''
    main flow of stacking.

    '''
    # stage 0
    # preprocessing
    train_,test_ = my_preprocessing(train,test)
    # adversarial validation
    train_ = adversarial_validation(train_,test_)
    target = train_['Survived']
    train_ = train_.drop('Survived',axis=1)
    # scaling
    #nm = my_normalizer()
    #nm.fit(train_)
    #train_ = nm.transform(train_)
    #test_ = nm.transform(test_)
    # get stage 0 
    X_train_st0 = train_
    X_test_st0 = test_
    print('Stage0, train shape: ',X_train_st0.shape)
    print('Stage0, test shape: ',X_test_st0.shape)
    print('')
    
    # stage0 -> stage1
    print('Stage1 '+'-'*50)
    X_train_st1, X_test_st1, cv_res_st1, total_elapsed_time_st1 = get_oofp(
        X_train_st0,
        X_test_st0,
        target,
        models=models['st1'],
        subsampling_num=5,
        subsampling_ratio=.5,
        )
    print('Stage 1 Average score: ',cv_res_st1['score'].mean())
    print('Stage 1 Total Running Time: ',round(total_elapsed_time_st1,1),'[sec]')
    print('')
    
    # add predictions from othres
    pass
    
    # stage1 -> stage2
    print('Stage2 '+'-'*50)
    X_train_st2, X_test_st2, cv_res_st2, total_elapsed_time_st2 = get_oofp(
        X_train_st1,
        X_test_st1,
        target,
        models=models['st2'],
        subsampling_num=1,
        subsampling_ratio=1.,
        )
    print('Stage 2 Average score: ',cv_res_st2['score'].mean())
    print('Stage 2 Total Running Time: ',round(total_elapsed_time_st2,1),'[sec]')
    print('')
    
    # stage2 -> stage3
    print('Stage3 '+'-'*50)
    X_train_st3, X_test_st3, cv_res_st3, total_elapsed_time_st3 = get_oofp(
        X_train_st2,
        X_test_st2,
        target,
        models=models['st3'],
        subsampling_num=1,
        subsampling_ratio=1.,
        )
    print('Stage 3 Average score: ',cv_res_st3['score'].mean())
    print('Stage 3 Total Running Time: ',round(total_elapsed_time_st3,1),'[sec]')
    print('')
    
    # stage3 -> stage final
    print('Stage final '+'-'*50)
    X_train_stf = pd.concat([X_train_st1, X_train_st2, X_train_st3],axis=1)
    X_test_stf = pd.concat([X_test_st1, X_test_st2, X_test_st3],axis=1)
    print('final train shape: ',X_train_stf.shape)
    print('final test shape: ',X_test_stf.shape)
    # define final model
    clf = models['stf']
    # define cv
    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=2020,
        )
    # cross validation
    cv_pred = cross_val_predict(
        estimator=clf,
        X=X_train_stf.values,
        y=target.values,
        cv=skf,
        )
    # eval
    # confusion matrix
    conf_mat = confusion_matrix(target.values,cv_pred)
    plot_confusion_matrix(conf_mat, list(target.unique()))
    # classification report
    cr = classification_report(
        y_true=target.values,
        y_pred=cv_pred,
        labels=target.unique(),
        target_names=['Survive','Death'],
        )
    print(cr)
    # refit
    clf.fit(X_train_stf.values,target.values)
    
    train_set = {
        'st0': X_train_st0,
        'st1': X_train_st1,
        'st2': X_train_st2,
        'st3': X_train_st3,
        'stf': X_train_stf,
        }
    test_set = {
        'st0': X_test_st0,
        'st1': X_test_st1,
        'st2': X_test_st2,
        'st3': X_test_st3,
        'stf': X_test_stf,
        }
    cv_res_set = {
        'st1': cv_res_st1,
        'st2': cv_res_st2,
        'st3': cv_res_st3,
        }
    total_elapsed_time_set = {
        'st1': total_elapsed_time_st1,
        'st2': total_elapsed_time_st2,
        'st3': total_elapsed_time_st3,   
        }
    return clf, train_set, target, test_set, cv_res_set, total_elapsed_time_set
