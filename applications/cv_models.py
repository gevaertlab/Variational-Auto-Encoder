""" cross validation pipeline """


from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch import isin
from applications.models import MODELS, data_summary
from applications.tasks.task_base import TaskBase
from utils.python_logger import get_logger
from .models import predict_with_model
from typing import Union
import numpy as np
import pandas as pd
LOGGER = get_logger()


def cv_predict_task(task: TaskBase,
                    X: Union[np.ndarray, pd.DataFrame],
                    Y: Union[np.ndarray, pd.DataFrame],
                    models="all",
                    fold=10,
                    hparam_dict={},
                    results=[],
                    tune_hparams=True,
                    verbose=True,
                    seed=9001):
    """ predict task using multiple models, with the same CV loop """
    prediction_models = MODELS[task.task_type]
    if models == "all":
        models = list(prediction_models.keys())
    elif isinstance(models, str):
        models = [models]
    assert isinstance(models, list)
    LOGGER.info(f"models used {models}")

    if results:
        result_dict = results
    else:
        result_dict = {}

    if verbose:
        LOGGER.info(
            f"Before transform: X shape = train:{np.array(X['train']).shape}, val:{np.array(X['val']).shape}; Y shape = train:{np.array(Y['train']).shape}, val:{np.array(Y['val']).shape}")

    X_processed, Y_processed = task.transform(X, Y)

    if verbose:
        # report data summary
        data_summary(X_processed, Y_processed, task.task_type)

    # stack X and Y
    if isinstance(X_processed['train'], pd.DataFrame):
        X_processed = pd.concat(
            [X_processed['train'], X_processed['val']], axis=0).reset_index()
    elif isinstance(X_processed['train'], np.ndarray):
        X_processed = np.concatenate(
            (X_processed['train'], X_processed['val']), axis=0)
    if isinstance(Y_processed['train'], pd.DataFrame):
        Y_processed = pd.concat(
            [Y_processed['train'], Y_processed['val']], axis=0).reset_index()
    elif isinstance(Y_processed['train'], np.ndarray):
        Y_processed = np.concatenate(
            (Y_processed['train'], Y_processed['val']), axis=0)

    for model_name in models:
        model_result = cv_predict_eval_with_model(task_type=task.task_type,
                                                  X=X_processed,
                                                  Y=Y_processed,
                                                  inverse_transform=task.inverse_transform,
                                                  model_name=model_name,
                                                  hparam_dict=hparam_dict,
                                                  tune_hparams=tune_hparams,
                                                  fold=fold,
                                                  verbose=verbose,
                                                  seed=seed)

        model_result_summary = [r[0] for r in model_result]
        model_metrics_dict = {}
        for key in model_result_summary[0].keys():
            model_metrics_dict[key] = [model_result_summary[i][key]
                                       for i in range(len(model_result_summary))]
        result_dict[model_name] = model_metrics_dict
    return result_dict


def cv_predict_eval_with_model(task_type: str,
                               X: Union[np.ndarray, pd.DataFrame],
                               Y: Union[np.ndarray, pd.DataFrame],
                               inverse_transform=None,
                               model_name: str = 'xgboost',
                               hparam_dict={},
                               tune_hparams=True,
                               fold=10,
                               verbose=True,
                               seed=9001):
    if verbose:
        LOGGER.info(f"======{model_name}======")
    assert task_type in ['classification', 'regression']

    # CV folds
    skf = StratifiedKFold(n_splits=fold,
                          shuffle=True,
                          random_state=seed)
    results = []
    for train_idx, test_idx in skf.split(X, Y):
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        elif isinstance(X, np.ndarray):
            X_train, X_test = X[train_idx], X[test_idx]
        if isinstance(Y, pd.DataFrame):
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        elif isinstance(Y, np.ndarray):
            Y_train, Y_test = Y[train_idx], Y[test_idx]
        print("train", Counter(Y_train), "test", Counter(Y_test))
        model_result = predict_with_model(task_type=task_type,
                                          X={"train": X_train, "val": X_test},
                                          Y={"train": Y_train, "val": Y_test},
                                          inverse_transform=inverse_transform,
                                          model_name=model_name,
                                          hparam_dict=hparam_dict,
                                          tune_hparams=tune_hparams,
                                          bootstrapping=False,
                                          verbose=verbose)
        results.append(model_result)
    return results
