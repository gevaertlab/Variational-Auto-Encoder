''' This file defines different tasks '''
''' A task consist of a name, task type (classification or regression), and a logic to map Xs to Ys '''
from typing import List, Dict
import pandas as pd
import os
import numpy as np


class TaskBase:
    SUPPORTED_TASKS_TYPES=['classification', 'regression']
    def __init__(self, name: str ='default task', task_type: str='classification'): #  -> None
        self.name = name
        assert task_type in self.SUPPORTED_TASKS_TYPES, f"Task type should be one of the following: {self.SUPPORTED_TASKS_TYPES}"
        self.task_type = task_type
        pass
    
    def getLabels(self, data_lst):
        raise NotImplementedError
    
    def merge(self, X, Y):
        ''' merge X and Y '''
        return X, Y
    
    
class TaskVolume(TaskBase):
    
    def __init__(self, name: str = "Volume size", task_type: str = 'regression', 
                 meta_data_path='/labs/gevaertlab/data/lung cancer/TCIA_LIDC/nodule_list.csv'):
        super(TaskVolume, self).__init__(name=name, task_type=task_type)
        self.meta_data_df = pd.read_csv(meta_data_path)
        self.meta_data_df['id'] = self.meta_data_df['scan'].astype(str) + '.' + self.meta_data_df['roi'].astype(str)
        pass
    
    def getLabels(self, data_lst):
        idx_rank = [self.getLabel(name) for name in data_lst]
        volume_origin_lst = self.meta_data_df['volume'].tolist()
        volume_lst = [volume_origin_lst[i] for i in idx_rank]
        return volume_lst
    
    def getIdx(self, data_name):
        return self.meta_data_df[self.meta_data_df['id'] == data_name].index.tolist()[0]
        
    def getLabel(self, data_name):
        return self.meta_data_df[self.meta_data_df['id'] == data_name]['volume'].values[0]
    
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

