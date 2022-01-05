from .task_base import TaskBase
import math
import numpy as np


class StfRG(TaskBase):

    def __init__(self,
                 name: str = 'stfrg',
                 task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)

    def transform(self, X, Y):
        # do nothing to X, and remove nan values in Y
        ifnan = np.array([self.check_nan(i) for i in Y['train']])
        X['train'] = X['train'][~ifnan]
        Y['train'] = np.array(Y['train'])[~ifnan]

        ifnan = np.array([self.check_nan(i) for i in Y['val']])
        X['val'] = X['val'][~ifnan]
        Y['val'] = np.array(Y['val'])[~ifnan]
        return X, Y

    def inverse_transform(self, X=None, Y=None):
        return X, Y

    @staticmethod
    def check_nan(input):
        if isinstance(input, str):
            if input == "Unknown":
                return True
            else:
                return False
        elif isinstance(input, float):
            if math.isnan(input):
                return True
        else:
            return True  # HACK
