''' This file contains sklearn models for downstream takss '''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# utils
from functools import partial
# models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
# hyperparameter tunning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
## Metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from typing import Dict

METRIC_DICT = {'MAE':mean_absolute_error,
               'MAPE':mean_absolute_percentage_error,
               'RMSE':partial(mean_squared_error, squared=False)}
## Dictionaries
RANDOM_DICT = {'random_state':9001}
NJOB_DICT = {'n_jobs':-1} # using all the processers

def modelEvaluation(y_true, y_pred, scoring_func_dict):
    return dict((key, func(y_true, y_pred)) for (key, func) in scoring_func_dict.items())

REGRESSION_MODELS = {'linear_regression':{'basemodel':partial(ElasticNet, **RANDOM_DICT), 
                                          'params':dict(alpha=[0.5, 1, 1.5],
                                                        l1_ratio=np.arange(0.2, 1, step=0.2))},
                     'random_forest':{'basemodel':partial(RandomForestRegressor, **RANDOM_DICT, **NJOB_DICT),
                                      'params':dict(n_estimators=[10, 100, 200])},
                     'svr':{'basemodel':SVR,
                            'params':dict(degree=[3, 5, 7], C=[0.1, 1, 10], epsilon=[0.01, 0.1, 1])},
                     'mlp':{'basemodel':partial(MLPRegressor, **RANDOM_DICT, early_stopping=True),
                            'params':dict(hidden_layer_sizes=[(100, 10), (100, 20), (200, 20), (400, 40)],
                                          max_iter=[2000])},
                     'gboost':{'basemodel':partial(GradientBoostingRegressor, **RANDOM_DICT),
                               'params':dict(learning_rate=[0.001, 0.01, 0.1, 0.5],
                                             n_estimators=[100, 200, 500])}
                     }

MODELS = {'regression':REGRESSION_MODELS, 'classification':None}

def predictWithModel(X: Dict, Y: Dict, task_type: str = 'classification', model_name: str = 'random_forest', verbose=True):
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
    model_meta = MODELS[task_type][model_name]
    basemodel = model_meta['basemodel']()
    clf = GridSearchCV(basemodel, model_meta['params'], verbose=int(verbose)*2) # verbose = 2 (a little more information) or 0
    clf.fit(X['train'], Y['train'])
    model = model_meta['basemodel'](**clf.best_params_).fit(X['train'], Y['train'])
    y_true, y_pred = Y['val'], model.predict(X['val'])
    return modelEvaluation(y_true, y_pred, METRIC_DICT)
    

def predictTask(X: Dict, Y: Dict, task_type: str = 'classification', models='all', verbose=True):
    prediction_models = MODELS[task_type]
    if models == 'all':
        models = list(prediction_models.keys())
    result_dict = {}
    for model_name in models:
        result_dict[model_name] = predictWithModel(X, Y, task_type, model_name, verbose=verbose)
    return result_dict
