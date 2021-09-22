""" provide backbone for label dataset. We can match a single label/batch of labels, save, load prematch labels, rematch label """

import functools
from configs.config_vars import BASE_DIR
from typing import List
import os
import numpy as np
import pylidc as dc
from multiprocessing import Pool, cpu_count
from utils.funcs import reorder, get_order, Timer, sort_dict


class Label:
    """ does not store labels ready for prediction, more like a label matching tool, please save label in dict of numpy arrays """

    def __init__(self,
                 name: str = 'volume',
                 log_dir=os.path.join(BASE_DIR, 'applications/logs/')):
        self.name = name
        self.log_dir = log_dir
        self.save_path = self._get_save_path()
        self._data = {}
        self.timer = Timer()
        self._advance_init()
        pass

    def _advance_init(self):
        # try to load labels
        if os.path.exists(self.save_path):
            self._data = self.load_labels()
        pass

    def _get_save_path(self):
        save_path = os.path.join(self.log_dir,
                                 "labels",
                                 f"Y_matched_{self.name}.npy")
        return save_path

    def match_label(self,
                    data_name: str):
        raise NotImplementedError

    def match_labels(self,
                     data_names: list,
                     n_jobs=cpu_count()):  # pool map
        self.timer()
        print(f"start matching {len(data_names)} labels ...")
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
        assert os.path.exists(self.save_path), "[Label] No file found"
        data = np.load(self.save_path, allow_pickle=True).item()
        return data

    @functools.singledispatch
    def get_labels(self, data_name: any):
        raise NotImplementedError

    @get_labels.register(str)
    def _(self, data_names: str):
        """
        get label from a single data_name
        """
        return self.get_labels([data_names])[0]

    @get_labels.register(list)
    def _(self, data_names: list):
        """ 
        get labels from 2 approaches 
        1. load
        2. match
        """
        # first match those that are not included in self._data
        new_names = [n for n in data_names if n not in self._data]
        if new_names:
            new_labels = self.match_labels(new_names)
            new_dict = {n: l for n, l in zip(new_names, new_labels)}
            self._data.update(new_dict)
            self._data = sort_dict(self._data)
            self.save_labels()
        # return
        return [self._data[key] for key in data_names]

    def reorder_labels(self, data_names: List, split='train', replace=True):
        """ reorder labels according to data_names """
        order = get_order(self.data_names[split], data_names)
        reordered_labels = reorder(self.labels[split], order)
        if replace:
            self.labels[split] = reordered_labels
            self.data_names[split] = data_names
        return reordered_labels
