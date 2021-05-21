from typing import List, Dict
import pandas as pd
import os
import numpy as np
import pylidc as dc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.funcs import reorder, get_order


class Label:

    def __init__(self,
                 name: str = 'volume',
                 log_dir='/labs/gevaertlab/users/yyhhli/code/vae/applications/logs/labels'):
        self.name = name
        self.log_dir = log_dir
        self.save_path = self._get_save_path()
        self.labels = None
        self.data_names = None

    def _get_save_path(self):
        save_path = os.path.join(self.log_dir,
                                 "labels",
                                 f"Y_matched_{self.name}.npy")
        return save_path

    def match_label(self, name):
        raise NotImplementedError

    def match_labels(self, data_names, n_jobs=cpu_count()):  # pool map
        print("Matching labels ...")
        pool = Pool(processes=n_jobs)
        return pool.map(self.match_label, data_names)

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

    def get_labels(self, data_names):
        """ 
        get labels from 3 approaches 
        1. load
        2. reorder
        3. match
        """
        # if stored, directly return it
        if self.labels is None:
            # try loading
            if os.path.exists(self.save_path):
                self.load_labels()
                # examine the matching
                if self.data_names != data_names:
                    self.reorder(data_names)

            # else: matching
            else:
                self.labels = self.match_labels(data_names)
            self.data_names = data_names
        return self.labels

    def reorder_labels(self, data_names: List, replace=True):
        """ reorder labels according to data_names """
        order = get_order(self.data_names, data_names)
        reordered_labels = reorder(self.labels, order)
        if replace:
            self.labels = reordered_labels
            self.data_names = data_names
        return self.labels


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
