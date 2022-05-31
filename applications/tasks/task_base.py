
import numpy as np


class TaskBase:
    SUPPORTED_TASKS_TYPES = ['classification', 'regression']

    def __init__(self,
                 name: str = 'default task',
                 task_type: str = 'classification'):  # -> None
        self.name = name
        assert task_type in self.SUPPORTED_TASKS_TYPES, f"Task type should be one of the following: {self.SUPPORTED_TASKS_TYPES}"
        self.task_type = task_type
        self.transform_dict = {}
        pass

    def transform(self, X, Y):
        """ 
        standardize X for default, and do nothing to Y 
        """
        # for X
        x_train_std, meta_dict = self.normalize(X['train'])
        x_val_std = (X['val'] - meta_dict['mean']) / meta_dict['std']
        self.transform_dict = meta_dict
        return {'train': x_train_std, 'val': x_val_std}, Y

    def transform_x(self, X):
        # for X
        x_train_std, meta_dict = self.normalize(X['train'])
        x_val_std = (X['val'] - meta_dict['mean']) / meta_dict['std']
        self.transform_dict = meta_dict
        return {'train': x_train_std, 'val': x_val_std}

    def inverse_transform_x(self, trans_x):
        if isinstance(trans_x, np.ndarray):
            return trans_x * self.transform_dict['std'] + self.transform_dict['mean']
        x_train = trans_x['train'] * \
            self.transform_dict['std'] + self.transform_dict['mean']
        x_val = trans_x['val'] * self.transform_dict['std'] + \
            self.transform_dict['mean']
        return {'train': x_train, 'val': x_val}

    def inverse_transform(self, X=None, Y=None):
        """ do inverse transformation for X or Y """
        if X is not None and Y is not None:
            return self.inverse_transform_x(X), Y
        elif X is not None:
            return self.inverse_transform_x(X)
        elif Y is not None:
            return Y
        else:
            return None

    def normalize(self, data_array):
        """
        Normalize each column to N(0, 1) distributed
        """
        meta_dict = {}
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        result_array = (data_array - mean) / std
        meta_dict['mean'] = mean
        meta_dict['std'] = std
        return result_array, meta_dict
