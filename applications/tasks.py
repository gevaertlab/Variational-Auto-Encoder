''' This file defines different tasks '''
''' A task consist of a name, task type (classification or regression), and a logic to map Xs to Ys '''




from typing import List, Dict
import pandas as pd
import os
import numpy as np
import pylidc as dc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
class TaskBase:
    SUPPORTED_TASKS_TYPES = ['classification', 'regression']

    def __init__(self, name: str = 'default task', task_type: str = 'classification'):  # -> None
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

    def match_label(self, name):
        raise NotImplementedError

    def match_labels(self, data_lst, n_jobs=cpu_count()):  # pool map
        print("Matching labels ...")
        pool = Pool(processes=n_jobs)
        return pool.map(self.match_label, data_lst)

    def merge(self, X, Y):
        ''' merge X and Y '''
        return X, Y

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

    def __init__(self, name: str = 'volume', task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.volume
        return value

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

    def match_label(self, data_name: str):  # -> int
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.malignancy
        return value


class TaskTexture(TaskBase):

    def __init__(self, name: str = 'texture', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)

    def match_label(self, data_name: str):  # -> int
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.texture
        return value


class TaskSpiculation(TaskBase):

    def __init__(self, name: str = 'spiculation', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)

    def match_label(self, data_name: str):  # -> int
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.spiculation
        return value


class TaskSubtlety(TaskBase):

    def __init__(self, name: str = 'subtlety', task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)

    def match_label(self, data_name: str):  # -> int
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.subtlety
        return value


# def debug():
#     tv = TaskVolume()
#     volume = tv.match_label('LIDC_IDRI-0008.129.jpeg')
#     print(volume)


# if __name__ == '__main__':
#     debug()


# class TaskVolume(TaskBase):

#     def __init__(self, name: str = "Volume size", task_type: str = 'regression',
#                  meta_data_path='/labs/gevaertlab/data/lung cancer/TCIA_LIDC/nodule_list.csv'):
#         super(TaskVolume, self).__init__(name=name, task_type=task_type)
#         self.meta_data_df = pd.read_csv(meta_data_path)
#         self.meta_data_df['id'] = self.meta_data_df['scan'].astype(str) + '.' + self.meta_data_df['roi'].astype(str)
#         pass

#     def getLabels(self, data_lst):
#         idx_rank = [self.getLabel(name) for name in data_lst]
#         volume_origin_lst = self.meta_data_df['volume'].tolist()
#         volume_lst = [volume_origin_lst[i] for i in idx_rank]
#         return volume_lst

#     def getIdx(self, data_name):
#         return self.meta_data_df[self.meta_data_df['id'] == data_name].index.tolist()[0]

#     def getLabel(self, data_name):
#         return self.meta_data_df[self.meta_data_df['id'] == data_name]['volume'].values[0]

"""
class TaskMalignancy(TaskBase):
    
    def __init__(self, name: str = "Volume size", task_type: str = 'regression', 
                meta_data_dir='/labs/gevaertlab/data/lung cancer/TCIA_LIDC/',
                diagnosis_file='tcia-diagnosis-data-2012-04-20.xls',
                nodule_lst_file='nodule_list.csv'):
        super(TaskMalignancy, self).__init__(name=name, task_type=task_type)
        self.nodule_lst_file = pd.read_csv(os.path.join(meta_data_dir, nodule_lst_file))
        self.diagnosis_file = pd.read_excel(os.path.join(meta_data_dir, diagnosis_file))
    pass

    def getLabels(self, data_lst):
        self.nodule_lst_file['id'] = self.nodule_lst_file['scan'].astype(str) + '.' + self.nodule_lst_file['roi'].astype(str)
        case_idx = [self.nodule_lst_file[self.nodule_lst_file['id'] == name]['case'].iloc[0] for name in data_lst]
        pids = self.case2PID(case_idx)
        roi_lst = self.nodule_lst_file['roi']
        diagnosis_lst = self.findDiagnosis(pids, roi_lst)
        return [int(d) for d in diagnosis_lst]
    
    def merge(self, X, Y):
        '''
        decide how to merge x and y based on the task
        '''
        # Y's should not be -1
        valid_idx = np.where(Y != -1)[0]
        return X[valid_idx], Y[valid_idx]
    
    def findDiagnosis(self, patient_id_lst, nodule_idx_lst):
        '''
        if found, return diatgnosis level
        if not found, return -1
        '''
        diagnosis_lst = []
        notfound = 0
        nodule_column_idx = [1, 4, 6, 8, 10, 12] # patient diagnosis, nodule 1, nodule 2 ... nodule 5
        for i in range(len(patient_id_lst)):
            pid, nodule_idx = patient_id_lst[i], nodule_idx_lst[i]
            df_filtered = self.diagnosis_file[self.diagnosis_file['TCIA Patient ID'] == pid]
            if len(df_filtered) > 0 and nodule_idx <= 5: # only nodule 1-5 has notations
                diagnosis = df_filtered.iloc[0, nodule_column_idx[nodule_idx]] # NOTE: this could be nan
                if np.isnan(diagnosis): 
                    diagnosis = -1
                    notfound += 1
            else:
                diagnosis = -1
                notfound += 1
            diagnosis_lst.append(diagnosis)
        print(f"{notfound} not found")
        return diagnosis_lst

    def case2PID(self, case_lst):
        return [self.__case2pid__(case) for case in case_lst]

    def __case2pid__(self, case: int):
        return f'LIDC-IDRI-{self.int2str(case)}'

    def int2str(self, input: int, pad=4):
        return str(input).zfill(pad)

"""
