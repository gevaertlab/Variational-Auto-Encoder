''' This file contains sklearn models for downstream takss '''

# utils
from functools import partial
from re import S
import numpy as np
import pandas as pd

# models
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, LogisticRegression
# Metrics
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, r2_score,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score, make_scorer)
# hyperparameter tunning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectFdr, f_regression, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from torch.utils.data.dataset import random_split
from utils.python_logger import get_logger

from applications.tasks.task_base import TaskBase

LOGGER = get_logger()


REGRESSION_METRIC_DICT = {'MAE': mean_absolute_error,
                          'MAPE': mean_absolute_percentage_error,
                          'RMSE': partial(mean_squared_error, squared=False),
                          'R2': r2_score}


CLASSIFICATION_METRIC_DICT = {'Accuracy': accuracy_score,
                              'F1': partial(f1_score, average='weighted'),
                              'Precision': partial(precision_score, average='weighted'),
                              'Recall': partial(recall_score, average='weighted'),
                              'AUROC': partial(roc_auc_score, average='weighted', multi_class='ovr'),
                              'AUPRC': partial(average_precision_score, average='weighted'),
                              }


# Dictionaries
RANDOM_DICT = {'random_state': 9001}
NJOB_DICT = {'n_jobs': -1}  # using all the processers


# model_dicts
# REGRESSION
li_reg = {'basemodel': partial(ElasticNet, l1_ratio=1, normalize=True, **RANDOM_DICT),
          'params': dict(alpha=[0.1, 0.5, 1])}

rfr = {'basemodel': partial(RandomForestRegressor, **RANDOM_DICT, **NJOB_DICT),
       'params': dict(n_estimators=[10, 100, 200])}

svr = {'basemodel': partial(SVR, max_iter=1000),  # efficiency
       'params': dict(degree=[3, 5, 7], C=[0.1, 1, 10], epsilon=[0.01, 0.1, 1])}

mlpr = {'basemodel': partial(MLPRegressor, **RANDOM_DICT, early_stopping=True, max_iter=2000),
        'params': dict(hidden_layer_sizes=[(100, 10),
                                           (100, 20),
                                           (200, 20),
                                           (400, 40)])}

gboost = {'basemodel': partial(GradientBoostingRegressor, **RANDOM_DICT),
          'params': dict(learning_rate=[0.01],
                         n_estimators=[200])}

REGRESSION_MODELS = {'linear_regression': li_reg,
                     'random_forest': rfr,
                     'svr': svr,
                     'mlp': mlpr,
                     }

# CLASSIFICATION
lr = {'basemodel': partial(LogisticRegression, penalty='l1', solver='saga', **RANDOM_DICT, **NJOB_DICT),
      'params': dict(C=[0.01, 0.1])}

knn = {'basemodel': partial(KNeighborsClassifier, **NJOB_DICT),
       'params': dict(n_neighbors=[3, 5, 7, 10], p=[1, 2, 5])}

svc = {'basemodel': partial(SVC, **RANDOM_DICT, probability=True, max_iter=1000),
       'params': dict(C=[0.01, 0.1, 1], degree=[1, 2, 3])}

rf = {'basemodel': partial(RandomForestClassifier, **RANDOM_DICT, **NJOB_DICT),
      'params': dict(n_estimators=[10, 100, 200])}

mlp = {'basemodel': partial(MLPClassifier, **RANDOM_DICT, early_stopping=True, max_iter=2000),
       'params': dict(hidden_layer_sizes=[(100, 10), (100, 20), (200, 20), (400, 40)])}


CLASSIFICATION_MODELS = {'logistic_regression': lr,
                         'k_nearest_neighbors': knn,
                         'svc': svc,
                         'random_forest': rf,
                         'mlp': mlp}

# preprocessing for pipeline
for modelname, modeldict in REGRESSION_MODELS.items():
    for param, paramvalue in modeldict['params'].items():
        modeldict['params'] = {}
        modeldict['params']["predictor__"+param] = paramvalue

for modelname, modeldict in CLASSIFICATION_MODELS.items():
    for param, paramvalue in modeldict['params'].items():
        modeldict['params'] = {}
        modeldict['params']["predictor__"+param] = paramvalue


MODELS = {'regression': REGRESSION_MODELS,
          'classification': CLASSIFICATION_MODELS}


def model_evaluation(y_true,
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
            if key == 'AUROC':
                if y_proba.shape[1] > 2:
                    y_score = y_proba
                else:
                    y_score = y_proba[:, 1]
                result_dict[key] = func(y_true=y_true,
                                        y_score=y_score)
            elif key == 'AUPRC':
                continue  # NOTE: not implemented at the moment
                if y_proba.shape[1] > 2:
                    continue
                result_dict[key] = func(y_true=y_true,
                                        y_score=y_proba[:, 1])
            else:
                result_dict[key] = func(y_true=y_true,
                                        y_pred=y_pred)
    return result_dict


def grid_cv_model(model, params, X, Y, scoring=None, verbose=1,
                  cv_params={"n_splits": 3, "shuffle": True, "random_state": 9001}):
    cv = KFold(**cv_params)
    clf = GridSearchCV(model,
                       params,
                       cv=cv,
                       scoring=scoring,
                       verbose=int(verbose)*2)  # verbose = 2 (a little more information) or 0
    clf.fit(X['train'], Y['train'])
    LOGGER.info(f"best parameters: {clf.best_params_}")
    return clf


def find_best_param(model_meta, X, Y, hparams={}, search=True, verbose=1):
    if search:
        basemodel = model_meta['basemodel']()
        clf = GridSearchCV(basemodel,
                           model_meta['params'],
                           verbose=int(verbose)*2)  # verbose = 2 (a little more information) or 0
        clf.fit(X['train'], Y['train'])
        LOGGER.info(f"best parameters: {clf.best_params_}")
        return clf.best_params_
    else:
        return hparams


def model_pipeline(model_base, base_params={}, best_params={}):
    # new step: feature selection
    # TODO: implement this
    model = Pipeline(
        [('scaler', StandardScaler()),
         #  ('selector', SelectFdr(score_func=f_regression, alpha=1e-2)),
         #  ('selector', SelectKBest(score_func=f_regression, k=500)),
         ('predictor', model_base(**base_params)),
         ], verbose=True)
    model.set_params(**best_params)
    return model


def predict_with_model(task_type: str,
                       X, Y,
                       inverse_transform=None,
                       model_name: str = 'random_forest',
                       hparam_dict={},
                       tune_hparams=True,
                       verbose=True):
    """
    NOTE: change preprocessing to be out side this function; avoid repeated operations
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
        LOGGER.info(f"======{model_name}======")
    assert task_type in ['classification', 'regression']

    # 1. hparams, either load or search or skip
    model_meta = MODELS[task_type][model_name]
    model = model_pipeline(model_base=model_meta['basemodel'])
    metrics_func_dict = REGRESSION_METRIC_DICT if task_type == "regression" else CLASSIFICATION_METRIC_DICT
    # has_key = model_name in hparam_dict.keys()
    # if has_key:
    #     hparams = hparam_dict[model_name]
    # else:
    #     hparams = {}
    if not tune_hparams:
        best_params = {}
    else:
        clf = grid_cv_model(model=model,
                            params=model_meta['params'],
                            X=X,
                            Y=Y,)

        best_params = clf.best_params_
        cv_results = {k: v for k, v in clf.cv_results_.items()
                      if k.endswith("score")}

    model = model_pipeline(model_base=model_meta['basemodel'],
                           best_params=best_params)
    # 2. train model
    model = model.fit(X['train'], Y['train'])

    # 3. evaluation
    if task_type == 'regression':
        if inverse_transform != None:
            y_pred = inverse_transform(Y=model.predict(X['val']))
        else:
            y_pred = model.predict(X['val'])
        y_true = Y['val']

        metrics = model_evaluation(y_true=y_true,
                                   y_pred=y_pred,
                                   y_proba=None,
                                   y_decision=None,
                                   scoring_func_dict=metrics_func_dict,
                                   task_type=task_type)
        # # print training results to see if the model can overfit
        # y_train_true, y_train_pred = Y['train'], \
        #     task.inverse_transform(Y=model.predict(x_trans['train']))
        # train_metrics = model_evaluation(y_true=y_train_true,
        #                                  y_pred=y_train_pred,
        #                                  y_proba=None,
        #                                  y_decision=None,
        #                                  scoring_func_dict=REGRESSION_METRIC_DICT,
        #                                  task_type=task.task_type)
        LOGGER.info(f"result for CV:{cv_results}")
        LOGGER.info(f"result for validation set:{metrics}")
        return metrics, y_pred, best_params

    elif task_type == 'classification':
        y_true = Y['val']
        y_pred = model.predict(X['val'])
        # multiclass, should not limit class
        y_proba = model.predict_proba(X['val'])
        # y_decision = model.decision_function(X['val'])
        y_decision = None
        metrics = model_evaluation(y_true=y_true,
                                   y_pred=y_pred,
                                   y_proba=y_proba,
                                   y_decision=y_decision,
                                   scoring_func_dict=metrics_func_dict,
                                   task_type=task_type)
        # print training results to see if the model can overfit
        # train_metrics = model_evaluation(y_true=Y['train'],
        #                                  y_pred=model.predict(
        #                                      x_trans['train']),
        #                                  y_proba=model.predict_proba(
        #     x_trans['train']),
        #     y_decision=None,
        #     scoring_func_dict=CLASSIFICATION_METRIC_DICT,
        #     task_type=task.task_type)
        LOGGER.info(f"result for CV:{cv_results}")
        LOGGER.info(f"result for validation set:{metrics}")
        return metrics, y_pred, {'y_proba': y_proba, 'y_decision': y_decision}, best_params


def predict_task(task: TaskBase,
                 X, Y,
                 models='all',
                 hparam_dict={},
                 results=[],
                 tune_hparams=True,
                 verbose=True):

    # 2. draw AUROC and AUPRC curve for classification
    prediction_models = MODELS[task.task_type]
    if models == 'all':
        models = list(prediction_models.keys())
    if results:
        result_dict, pred_dict, pred_stats, best_hparams = results
        pred_dict['true'] = Y['val']
    else:
        result_dict, pred_dict, pred_stats, best_hparams = {}, {}, {}, {}
        pred_dict['true'] = Y['val']

    # preprocessing with task.transform
    if verbose:
        LOGGER.info(
            f"Before transform: X shape = train:{np.array(X['train']).shape}, val:{np.array(X['val']).shape}; Y shape = train:{np.array(Y['train']).shape}, val:{np.array(Y['val']).shape}")

    X_processed, Y_processed = task.transform(X, Y)

    if verbose:
        # report data summary

        data_summary = f"X shape = train:{X_processed['train'].shape}, val:{X_processed['val'].shape}; Y shape = train:{Y_processed['train'].shape}, val:{Y_processed['val'].shape}"
        if task.task_type == "classification":
            vtrain, ctarin = np.unique(
                np.array(Y_processed['train']), return_counts=True)
            train_count = pd.DataFrame(np.array((vtrain, ctarin)).T,
                                       columns=["value", "count"])
            vval, cval = np.unique(
                np.array(Y_processed['val']), return_counts=True)
            val_count = pd.DataFrame(np.array((vval, cval)).T,
                                     columns=["value", "count"])
            data_summary += "\n" + \
                f"Y classes = train: \n{train_count}; val: \n{val_count}"
        LOGGER.info(data_summary)

    for model_name in models:
        model_result = predict_with_model(task_type=task.task_type,
                                          X=X_processed, Y=Y_processed,
                                          model_name=model_name,
                                          hparam_dict=hparam_dict,
                                          tune_hparams=tune_hparams,
                                          verbose=verbose)
        if task.task_type == "classification":
            result_dict[model_name], \
                pred_dict[model_name], \
                pred_stats[model_name], \
                best_hparams[model_name] = model_result
        elif task.task_type == "regression":
            result_dict[model_name], \
                pred_dict[model_name], \
                best_hparams[model_name] = model_result

        # update saving
        for item in results:
            item.save()
    return result_dict, pred_dict, pred_stats, best_hparams
