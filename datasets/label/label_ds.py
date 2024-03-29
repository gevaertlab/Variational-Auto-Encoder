""" provide backbone for label dataset. We can match a single label/batch of labels, save, load prematch labels, rematch label """

import functools

from tqdm import tqdm
from configs.config_vars import BASE_DIR
from typing import Any, List, Union
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from utils.funcs import reorder, get_order, sort_dict
from utils.timer import Timer


class Label:
    """ 
    does not store labels ready for prediction, 
    more like a label matching tool, please save label in dict of numpy arrays 
    """

    def __init__(self,
                 name: str = 'volume',
                 dataset_name: Union[str, None] = None,
                 log_dir=os.path.join(BASE_DIR, 'applications/logs/'),
                 **kwds):
        self.name = name
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.save_path = self._get_save_path()
        self._data = {}
        self.timer = Timer((__file__, self.__class__.__name__))
        self._advance_init()
        pass

    def _advance_init(self):
        # try to load labels
        if os.path.exists(self.save_path):
            self._data = self.load_labels()
        pass

    def _get_save_path(self):
        if self.dataset_name:
            save_dir = os.path.join(self.log_dir,
                                    "labels",
                                    self.dataset_name)
        else:
            save_dir = os.path.join(self.log_dir, "labels")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir,
                                 f"Y_matched_{self.name}.npy")
        return save_path

    def match_label(self,
                    data_name: str):
        raise NotImplementedError

    def match_labels(self,
                     data_names: list,
                     multiprocess=False,
                     n_jobs=cpu_count()):  # pool map
        self.timer()
        print(f"start matching {len(data_names)} labels ...")
        if multiprocess:
            try:
                pool = Pool(processes=n_jobs)
                result = pool.map(self.match_label, data_names)
            except Exception as e:
                result = [self.match_label(dn) for dn in tqdm(data_names)]
        else:
            result = [self.match_label(dn) for dn in tqdm(data_names)]
        self.timer("matching labels")
        return result

    def save_labels(self):
        """ save matched label as npy file """
        # data = {'data_names': self.data_names,
        #         'labels': self.labels}
        np.save(self.save_path, self._data)
        print(f"saved to {self.save_path}")
        pass

    def load_labels(self):
        assert os.path.exists(self.save_path), "[Label] No file found"
        data = np.load(self.save_path, allow_pickle=True).item()
        return data

    @functools.singledispatch
    def get_labels(self, data_name: list):
        """ 
        get labels from 2 approaches 
        1. load
        2. match
        """
        # first match those that are not included in self._data
        new_names = [n for n in data_name if n not in self._data]
        if new_names:
            new_labels = self.match_labels(new_names)
            new_dict = {n: l for n, l in zip(new_names, new_labels)}
            self._data.update(new_dict)
            self._data = sort_dict(self._data)
            self.save_labels()
        # return
        return [self._data[key] for key in data_name]

    @get_labels.register(str)
    def _(self, data_name: str):
        """
        get label from a single data_name
        """
        return self.get_labels([data_name])[0]

    def reorder_labels(self, data_names: List, split='train', replace=True):
        """ reorder labels according to data_names """
        order = get_order(self.data_names[split], data_names)
        reordered_labels = reorder(self.labels[split], order)
        if replace:
            self.labels[split] = reordered_labels
            self.data_names[split] = data_names
        return reordered_labels
