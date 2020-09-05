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
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./my_modules/')
from utils import LOCAL_TRAIN_RATIO
from utils import plot_confusion_matrix
from mytorch import DEVICE, MyDataset, MyNormalizer, my_mlp_trainer


def get_evaluate(y_test, predict_proba, clf_threshold):
    fpr, tpr, _ = metrics.roc_curve(y_test, predict_proba)
    auc = metrics.auc(fpr, tpr)
    predict = [0 if i < clf_threshold else 1 for i in predict_proba]
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    return auc, precision, recall


def my_cross_validation(
    cv_conf: dict,
    train: pd.DataFrame,
    target: str,
    features: list,
    dpath_to_checkpoints: str,
    verbose=True,
    ):
    # init
    # build splitter
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2020,
        )
    # setup result
    if cv_conf['name']=='lgb':
        res = {
            'eval': pd.DataFrame(),
            'pred': train,
            'feature_importance': pd.DataFrame(index=features),
            'learning_history': {},
            'model_dicts': {},
        }
    elif cv_conf['name']=='mlp':
        res = {
            'eval': pd.DataFrame(),
            'pred': train,
        }
    else:
        raise Exception(f'Not found {cv_conf["name"]}')

    # iterate over folds
    for i, (cv_train_index, cv_test_index) in enumerate(skf.split(train[features],train[target])):
        if verbose:
            print(f'{i+1}th fold')
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
        
        if cv_conf['name']=='lgb':
            # make dataset
            lgb_train = lgb.Dataset(local_train[features], local_train[target])
            lgb_val = lgb.Dataset(local_val[features], local_val[target])
            # training
            evals_result = {}
            clf = lgb.train(
                params=cv_conf['params'],
                train_set=lgb_train,
                valid_sets=[lgb_val],
                valid_names=['val'],
                evals_result=evals_result,
                )
            # save training result
            res['feature_importance'].loc[:,f'{i+1}th'] = clf.feature_importance(importance_type='gain')
            res['learning_history'][f'{i+1}th'] = evals_result
            res['models'][f'{i+1}th'] = clf
            # gert prediction
            predict_proba = clf.predict(cv_test[features], num_iteration=clf.best_iteration)
        
        elif cv_conf['name']=='mlp':
            # reset net weights
            if i==0:
                fpath_to_init_weight = os.path.join(dpath_to_checkpoints,'init_weights.pt')
                torch.save(cv_conf['training_params']['net'].state_dict(),fpath_to_init_weight)
            else:
                cv_conf['training_params']['net'].load_state_dict(torch.load(fpath_to_init_weight))
            # build normalizer
            nm = MyNormalizer()
            nm.fit(local_train[features])
            # build tensorboard log writer
            dpath_to_logs = os.path.join(dpath_to_checkpoints,'logs',f'{i+1}th_fold')
            for sdir,_,files in os.walk(dpath_to_logs):
                if files:
                    for f in files:
                        os.remove(os.path.join(sdir,f))
            train_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'train'))
            val_writer = SummaryWriter(log_dir=os.path.join(dpath_to_logs,'val'))
            # setup file path to model parameters
            fpath_to_model_state_dict = os.path.join(dpath_to_checkpoints,'models',f'{i+1}th_model.pt')
            if i==0:
                parent = os.path.split(fpath_to_model_state_dict)[0]
                if not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
            if os.path.exists(fpath_to_model_state_dict):
                os.remove(fpath_to_model_state_dict)
            # run training
            net = my_mlp_trainer(
                normalizer=nm,
                X_train=local_train[features],
                y_train=local_train[target],
                X_val=local_val[features],
                y_val=local_val[target],
                fpath_to_model_state_dict=fpath_to_model_state_dict,
                train_writer=train_writer,
                val_writer=val_writer,
                **cv_conf['training_params'],
                )
            # get prediction
            net.eval()
            with torch.no_grad():
                test_batch = nm.transform(cv_test[features])
                test_batch = torch.from_numpy(test_batch.values).float().to(DEVICE)
                predict_proba = net(test_batch)
                predict_proba = predict_proba.to('cpu').detach().numpy()
        
        # save prediction
        train.loc[cv_test_index,'pred'] = predict_proba
        # evaluation
        auc, precision, recall = get_evaluate(cv_test[target].values, predict_proba, cv_conf['clf_threshold'])
        # save eval result
        res['eval'].loc[f'{i+1}th','auc'] = auc
        res['eval'].loc[f'{i+1}th','precision'] = precision
        res['eval'].loc[f'{i+1}th','recall'] = recall
        res['eval'].loc[f'{i+1}th','f'] = (2*precision*recall)/(recall+precision)
        
        train.loc[cv_test_index,'cv_fold'] = i+1

    return res


def agg_cv_models(
    train,
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
    # # stage 0
    # # preprocessing
    # train_,test_ = my_preprocessing(train,test)
    # # adversarial validation
    # train_ = adversarial_validation(train_,test_)
    # target = train_['Survived']
    # train_ = train_.drop('Survived',axis=1)
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
