''' This file defines different tasks '''
''' A task consist of a name, task type (classification or regression), and a logic to map Xs to Ys '''


# TODO: inplement label class and then task class




from typing import List, Dict
import pandas as pd
import os
import numpy as np
import pylidc as dc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.funcs import reorder, get_order
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

    def get_scan_ann_from_file_name(self, data_name: str):
        patient_id, ann_id = data_name.split('.')[0], data_name.split('.')[1]
        scan = dc.query(dc.Scan).filter(
            dc.Scan.patient_id == patient_id).first()
        ann = dc.query(dc.Annotation).filter(
            dc.Annotation.id == ann_id).first()
        return scan, ann

    def transform(self, X, Y):
        """ standardize X for default, and do nothing to Y """
        # for X
        x_train_std, meta_dict = self.normalize(X['train'])
        x_val_std = (X['val'] - meta_dict['mean']) / meta_dict['std']
        self.transform_dict = meta_dict
        return {'train': x_train_std, 'val': x_val_std}, Y

    def inverse_transform(self, X=None, Y=None):
        """ do inverse transformation for X or Y """
        if X is not None and Y is not None:
            return X, Y
        elif X is not None:
            return X
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


class TaskVolume(TaskBase):

    def __init__(self,
                 name: str = 'volume',
                 task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)

    def transform(self, X, Y):  # HACK: hard code X and Y's keys
        """ Standardize X and then log transform Y """
        # for X
        x_train_std, meta_dict = self.normalize(X['train'])
        x_val_std = (X['val'] - meta_dict['mean']) / meta_dict['std']
        self.transform_dict = meta_dict
        # for Y
        y_trian_log = np.log(Y['train'])
        y_val_log = np.log(Y['val'])
        # returned a new dictionary, different from the previous one
        return {'train': x_train_std, 'val': x_val_std}, \
               {'train': y_trian_log, 'val': y_val_log}

    def inverse_transform(self, X=None, Y=None):
        """ X -> stadardized, Y -> log taken """
        if X is not None and Y is not None:
            new_x, new_y = {}, {}
            # for X
            for key, value in X.items():
                new_x[key] = value * self.transform_dict['std'] + \
                    self.transform_dict['mean']
            # for Y
            for key, value in Y.items():
                new_y[key] = np.exp(value)
            return new_x, new_y
        elif X is not None:
            new_x = {}
            for key, value in X.items():
                new_x[key] = value * self.transform_dict['std'] + \
                    self.transform_dict['mean']
            return new_x
        elif Y is not None:
            new_y = {}
            for key, value in Y.items():
                new_y[key] = np.exp(value)
            return new_y
        else:
            return None


class TaskMalignancy(TaskBase):

    def __init__(self, name: str = 'malignancy', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)


class TaskTexture(TaskBase):

    def __init__(self, name: str = 'texture', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)


class TaskSpiculation(TaskBase):

    def __init__(self, name: str = 'spiculation', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)


class TaskSubtlety(TaskBase):

    def __init__(self, name: str = 'subtlety', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)


# def debug():
#     tv = TaskVolume()
#     tv.get_labels()
#     print(volume)


# if __name__ == '__main__':
#     debug()
