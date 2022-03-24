
import numpy as np
import pylidc as dc
from .task_base import TaskBase


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
