''' This file contains sklearn models for downstream takss '''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# utils
from functools import partial
# models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
# hyperparameter tunning
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from typing import Dict


REGRESSION_METRIC_DICT = {'MAE': mean_absolute_error,
                          'MAPE': mean_absolute_percentage_error,
                          'RMSE': partial(mean_squared_error, squared=False)}


CLASSIFICATION_METRIC_DICT = {'Accuracy': accuracy_score,
                              'F1': partial(f1_score, average='weighted'),
                              'Precision': partial(precision_score, average='weighted'),
                              'Recall': partial(recall_score, average='weighted'),
                              'AUROC': partial(roc_auc_score, average='weighted', multi_class='ovr'),
                              #   'AUPRC': partial(average_precision_score, average='weighted'),
                              }


# Dictionaries
RANDOM_DICT = {'random_state': 9001}
NJOB_DICT = {'n_jobs': -1}  # using all the processers


# model_dicts
# REGRESSION
li_reg = {'basemodel': partial(ElasticNet, **RANDOM_DICT),
          'params': dict(alpha=[0.5, 1, 1.5],
                         l1_ratio=np.arange(0.2, 1, step=0.2))}

rfr = {'basemodel': partial(RandomForestRegressor, **RANDOM_DICT, **NJOB_DICT),
       'params': dict(n_estimators=[10, 100, 200])}

svr = {'basemodel': SVR,
       'params': dict(degree=[3, 5, 7], C=[0.1, 1, 10], epsilon=[0.01, 0.1, 1])}

mlpr = {'basemodel': partial(MLPRegressor, **RANDOM_DICT, early_stopping=True),
        'params': dict(hidden_layer_sizes=[(100, 10), (100, 20), (200, 20), (400, 40)],
                       max_iter=[2000])}

gboost = {'basemodel': partial(GradientBoostingRegressor, **RANDOM_DICT),
          'params': dict(learning_rate=[0.01],
                         n_estimators=[200])}

REGRESSION_MODELS = {'linear_regression': li_reg,
                     'random_forest': rfr,
                     'svr': svr,
                     'mlp': mlpr,
                     }

# CLASSIFICATION
lr = {'basemodel': partial(LogisticRegression, penalty='elasticnet', solver='saga', **RANDOM_DICT, **NJOB_DICT),
      'params': dict(C=[0.01, 0.1], l1_ratio=[0, 0.3, 0.5, 0.7, 1])}

knn = {'basemodel': partial(KNeighborsClassifier, **NJOB_DICT),
       'params': dict(n_neighbors=[3, 5, 7, 10], p=[1, 1, 5, 2])}

svc = {'basemodel': partial(SVC, **RANDOM_DICT, probability=True),
       'params': dict(C=[0.01, 0.1, 1], degree=[1, 2, 3])}

rf = {'basemodel': partial(RandomForestClassifier, **RANDOM_DICT, **NJOB_DICT),
      'params': dict(n_estimators=[10, 100, 200])}

mlp = {'basemodel': partial(MLPClassifier, **RANDOM_DICT, early_stopping=True),
       'params': dict(hidden_layer_sizes=[(100, 10), (100, 20), (200, 20), (400, 40)], max_iter=[2000])}


CLASSIFICATION_MODELS = {'logistic_regression': lr,
                         'k_nearest_neighbors': knn,
                         'svc': svc,
                         'random_forest': rf,
                         'mlp': mlp}


MODELS = {'regression': REGRESSION_MODELS,
          'classification': CLASSIFICATION_MODELS}


def modelEvaluation(y_true,
                    y_pred,
                    y_proba=None,
                    y_decision=None,
                    scoring_func_dict=REGRESSION_METRIC_DICT,
                    task_type='regression'):
    if task_type == "regression":
        return dict((key, func(y_true, y_pred)) for (key, func) in scoring_func_dict.items())
    elif task_type == "classification":
        result_dict = {}
        for (key, func) in scoring_func_dict.items():
            if key in ['AUROC']:
                result_dict[key] = func(y_true=y_true,
                                        y_score=y_proba)
            # elif key == 'AUPRC':
            #     result_dict[key] = func(y_true=y_true,
            #                             y_score=y_decision)
            else:
                result_dict[key] = func(y_true=y_true,
                                        y_pred=y_pred)
    return result_dict


def find_best_param(model_meta, X, Y, hparams={}, search=True, verbose=1):
    if search:
        basemodel = model_meta['basemodel']()
        clf = GridSearchCV(basemodel, model_meta['params'], verbose=int(
            verbose)*2)  # verbose = 2 (a little more information) or 0
        clf.fit(X['train'], Y['train'])
        print(f"best parameters: {clf.best_params_}")
        return clf.best_params_
    else:
        return hparams


def predictWithModel(X: Dict,
                     Y: Dict,
                     task_type: str = 'classification',
                     model_name: str = 'random_forest',
                     hparam_dict={},
                     verbose=True):
    """
    Make predictions with one type of model, hparams are tuned, model results are returned
    Args:
        X (Dict): input X, has key 'train' and 'val'
        Y (Dict): input Y, has key 'train' and 'val'
        task_type (str, optional): type of task, should be either classification or regression. Defaults to 'classification'.
        model_name (str, optional): name of the model. Defaults to 'random_forest'.
        verbose (bool, optional): whether to output some progress. Defaults to True.

    Returns:
        Dict: metrics
    """
    if verbose:
        print(f"======{model_name}======")
    assert task_type in ['classification', 'regression']

    # 1. hparams, either load or search
    model_meta = MODELS[task_type][model_name]
    has_key = model_name in hparam_dict.keys()
    if has_key:
        hparams = hparam_dict[model_name]
    else:
        hparams = {}
    best_params = find_best_param(model_meta, X, Y,
                                  hparams=hparams,
                                  search=not has_key)

    # 2. train model
    model = model_meta['basemodel'](
        **best_params).fit(X['train'], Y['train'])

    # 3. evaluation
    if task_type == 'regression':
        y_true, y_pred = Y['val'], model.predict(X['val'])
        metrics = modelEvaluation(y_true=y_true,
                                  y_pred=y_pred,
                                  y_proba=None,
                                  y_decision=None,
                                  scoring_func_dict=REGRESSION_METRIC_DICT,
                                  task_type=task_type)
        # print training results to see if the model can overfit
        train_metrics = modelEvaluation(y_true=Y['train'],
                                        y_pred=model.predict(X['train']),
                                        y_proba=None,
                                        y_decision=None,
                                        scoring_func_dict=REGRESSION_METRIC_DICT,
                                        task_type=task_type)
        print("Result for training set:", train_metrics)
        return metrics, y_pred, best_params

    elif task_type == 'classification':
        y_true = Y['val']
        y_pred = model.predict(X['val'])
        # multiclass, should not limit class
        y_proba = model.predict_proba(X['val'])
        # y_decision = model.decision_function(X['val'])
        y_decision = None
        metrics = modelEvaluation(y_true=y_true,
                                  y_pred=y_pred,
                                  y_proba=y_proba,
                                  y_decision=y_decision,
                                  scoring_func_dict=CLASSIFICATION_METRIC_DICT,
                                  task_type=task_type)
        # print training results to see if the model can overfit
        train_metrics = modelEvaluation(y_true=Y['train'],
                                        y_pred=model.predict(X['train']),
                                        y_proba=model.predict_proba(
                                            X['train']),
                                        y_decision=None,
                                        scoring_func_dict=CLASSIFICATION_METRIC_DICT,
                                        task_type=task_type)
        print("Result for training set:", train_metrics)
        return metrics, y_pred, {'y_proba': y_proba, 'y_decision': y_decision}, best_params


def predictTask(X: Dict, Y: Dict,
                task_type: str = 'classification',
                models='all',
                hparam_dict={},
                results=[],
                verbose=True):
    prediction_models = MODELS[task_type]
    if models == 'all':
        models = list(prediction_models.keys())
    if results:
        result_dict, pred_dict, pred_stats, best_hparams = results
        pred_dict['true'] = Y['val']
    else:
        result_dict, pred_dict, pred_stats, best_hparams = {}, {}, {}, {}
        pred_dict['true'] = Y['val']

    for model_name in models:
        model_result = predictWithModel(X, Y, task_type,
                                        model_name,
                                        hparam_dict=hparam_dict,
                                        verbose=verbose)
        if task_type == "classification":
            result_dict[model_name], \
                pred_dict[model_name], \
                pred_stats[model_name], \
                best_hparams[model_name] = model_result
        elif task_type == "regression":
            result_dict[model_name], \
                pred_dict[model_name], \
                best_hparams[model_name] = model_result

        # update saving
        for item in results:
            item.save()
    return result_dict, pred_dict, pred_stats, best_hparams
