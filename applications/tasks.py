''' This file defines different tasks '''
# A task consist of a name, task type (classification or regression), and a logic to map Xs to Ys


import numpy as np
import pylidc as dc


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
        """ standardize X for default, and do nothing to Y """
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


class LNDbTaskVolume(TaskBase):

    def __init__(self,
                 name: str = 'lndb_volmne',
                 task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)


class LNDbTaskTexture(TaskBase):

    def __init__(self,
                 name: str = 'lndb_texture',
                 task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)


class LIDCTaskBase(TaskBase):

    def __init__(self,
                 name: str = 'default task',
                 task_type: str = 'classification'):  # -> None
        super(LIDCTaskBase, self).__init__(name=name, task_type=task_type)
        pass

    def get_scan_ann_from_file_name(self, data_name: str):
        patient_id, ann_id = data_name.split('.')[0], data_name.split('.')[1]
        scan = dc.query(dc.Scan).filter(
            dc.Scan.patient_id == patient_id).first()
        ann = dc.query(dc.Annotation).filter(
            dc.Annotation.id == ann_id).first()
        return scan, ann


class TaskVolume(LIDCTaskBase):

    def __init__(self,
                 name: str = 'volume',
                 task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)

    def transform(self, X, Y):  # HACK: hard code X and Y's keys
        """ Standardize X and then log transform Y """
        trans_x = self.transform_x(X)
        trans_y = self.transform_y(Y)
        return trans_x, trans_y

    def inverse_transform(self, X=None, Y=None):
        if X is None and Y is None:
            return None

        if X is not None:
            X = self.inverse_transform_x(X)
        if Y is not None:
            Y = self.inverse_transform_y(Y)

        if X is None:
            return Y
        elif Y is None:
            return X
        else:
            return X, Y

    def transform_x(self, X):
        return super().transform_x(X)

    def inverse_transform_x(self, trans_x):
        return super().inverse_transform_x(trans_x)

    def transform_y(self, Y):
        # for Y
        y_train_log = np.log(Y['train'])
        y_val_log = np.log(Y['val'])
        return {'train': y_train_log, "val": y_val_log}

    def inverse_transform_y(self, trans_y):
        if isinstance(trans_y, np.ndarray):
            return np.exp(trans_y)
        y_train = np.exp(trans_y['train'])
        y_val = np.exp(trans_y['val'])
        return {'train': y_train, "val": y_val}


class TaskMalignancy(LIDCTaskBase):  # HACK

    def __init__(self, name: str = 'malignancy', task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)


class TaskTexture(LIDCTaskBase):

    def __init__(self, name: str = 'texture', task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)


class TaskSpiculation(LIDCTaskBase):

    def __init__(self, name: str = 'spiculation', task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)


class TaskSubtlety(LIDCTaskBase):

    def __init__(self, name: str = 'subtlety', task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)
