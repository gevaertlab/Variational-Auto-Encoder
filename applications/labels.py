from typing import List, Dict
import pandas as pd
import os
import numpy as np
import pylidc as dc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.funcs import reorder, get_order, Timer


class Label:

    def __init__(self,
                 name: str = 'volume',
                 log_dir='/labs/gevaertlab/users/yyhhli/code/vae/applications/logs/'):
        self.name = name
        self.log_dir = log_dir
        self.save_path = self._get_save_path()
        self.labels = {}
        self.data_names = {}
        self.timer = Timer()

    def _get_save_path(self):
        save_path = os.path.join(self.log_dir,
                                 "labels",
                                 f"Y_matched_{self.name}.npy")
        return save_path

    def get_scan_ann_from_file_name(self, data_name: str):
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        patient_id, ann_id = data_name.split('.')[0], data_name.split('.')[1]
        scan = dc.query(dc.Scan).filter(
            dc.Scan.patient_id == patient_id).first()
        ann = dc.query(dc.Annotation).filter(
            dc.Annotation.id == ann_id).first()
        return scan, ann

    def match_label(self, data_name):
        raise NotImplementedError

    def match_labels(self, data_names, n_jobs=cpu_count()):  # pool map
        self.timer()
        print("start matching labels ...")
        pool = Pool(processes=n_jobs)
        result = pool.map(self.match_label, data_names)
        self.timer("matching labels")
        return result

    def save_labels(self):
        """ save matched label as npy file """
        data = {'data_names': self.data_names,
                'labels': self.labels}
        np.save(self.save_path, data)
        print(f"saved to {self.save_path}")
        pass

    def load_labels(self):
        data = np.load(self.save_path, allow_pickle=True).item()
        self.labels = data['labels']
        self.data_names = data['data_names']
        return data

    def get_labels(self, data_names, split='train'):
        """ 
        get labels from 3 approaches 
        1. load
        2. reorder
        3. match
        return: Y = {'train': ..., 'val': ...}
        the returned Y should have data_names 
        exactly matching the query data_names
        """
        if split not in self.labels:
            # loading
            if os.path.exists(self.save_path):
                self.load_labels()
            # if can't: matching
            # NOTE: func <match_labels> will output exactly order of data_names
            else:
                self.labels[split] = self.match_labels(data_names)
                self.data_names[split] = data_names
        # at this point, self.label should be initialized
        # see if we need reorder
        if self.data_names[split] != data_names:
            # NOTE: if data_names not match, and query data_names is shorter,
            # which means we can reorder to fix it, else, just re-matching
            if len(data_names) <= len(self.data_names[split]):
                reordered_labels = self.reorder_labels(data_names,
                                                       split,
                                                       replace=False)
                # assertion
                assert len(data_names) == len(reordered_labels), \
                    "label matching failed."
                return reordered_labels
            else:
                self.labels[split] = self.match_labels(data_names)
                self.data_names[split] = data_names
        # assertion
        assert len(data_names) == len(self.labels[split]) and \
            self.data_names[split] == data_names, \
            "label matching failed."

        return self.labels[split]

    def reorder_labels(self, data_names: List, split='train', replace=True):
        """ reorder labels according to data_names """
        order = get_order(self.data_names[split], data_names)
        reordered_labels = reorder(self.labels[split], order)
        if replace:
            self.labels[split] = reordered_labels
            self.data_names[split] = data_names
        return reordered_labels


class LabelVolume(Label):

    def __init__(self,
                 name: str = 'volume'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.volume
        return value


class LabelMalignancy(Label):

    def __init__(self,
                 name: str = 'malignancy'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> int
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.malignancy
        return value


class LabelTexture(Label):

    def __init__(self,
                 name: str = 'texture'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.texture
        return value


class LabelSpiculation(Label):

    def __init__(self,
                 name: str = 'spiculation'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.spiculation
        return value


class LabelSubtlety(Label):

    def __init__(self,
                 name: str = 'subtlety'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.subtlety
        return value
