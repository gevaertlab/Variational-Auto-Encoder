from .task_base import TaskBase
import math
import numpy as np


class StfRG(TaskBase):

    def __init__(self,
                 name: str = 'stfrg',
                 task_type: str = 'classification',
                 nan_list: list = ["NA", "Unknown"]):
        self.nan_list = nan_list
        super().__init__(name=name, task_type=task_type)

    def transform(self, X, Y):
        # do nothing to X, and remove nan values in Y
        # NOTE: watch for X type and Y type, convert to numpy

        if not isinstance(X['train'], np.ndarray):
            X['train'] = np.array(X['train'])
        if not isinstance(X['val'], np.ndarray):
            X['val'] = np.array(X['val'])

        ifnan = np.array([self.check_nan(i) for i in Y['train']])
        X['train'] = X['train'][~ifnan]
        Y['train'] = np.asarray(Y['train'])[~ifnan]

        ifnan = np.array([self.check_nan(i) for i in Y['val']])
        X['val'] = X['val'][~ifnan]
        Y['val'] = np.asarray(Y['val'])[~ifnan]
        return X, Y

    def inverse_transform(self, X=None, Y=None):
        return X, Y

    def check_nan(self, input):
        # mostly don't assume nan
        if isinstance(input, str):
            if input in self.nan_list:
                return True
            else:
                return False
        elif isinstance(input, float):
            if math.isnan(input):
                return True
        else:
            return False  # HACK
