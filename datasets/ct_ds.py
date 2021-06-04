""" 
CT dataset
The CT dataset that can has several functionalities:
    1. load CT <numpy images of HU units>;
    2. position of nodule <centroid>;
    3. name;
"""


import os
import re
from time import time
import SimpleITK as sitk
import pandas as pd
import pydicom


class bidict(dict):
    """ 
    bidirectional dictionary 
        Usage:
    bd = bidict({'a': 1, 'b': 2})  
    print(bd)                     # {'a': 1, 'b': 2}                 
    print(bd.inverse)             # {1: ['a'], 2: ['b']}
    bd['c'] = 1                   # Now two keys have the same value (= 1)
    print(bd)                     # {'a': 1, 'c': 1, 'b': 2}
    print(bd.inverse)             # {1: ['a', 'c'], 2: ['b']}
    del bd['c']
    print(bd)                     # {'a': 1, 'b': 2}
    print(bd.inverse)             # {1: ['a'], 2: ['b']}
    del bd['a']
    print(bd)                     # {'b': 2}
    print(bd.inverse)             # {2: ['b']}
    bd['b'] = 3
    print(bd)                     # {'b': 3}
    print(bd.inverse)             # {2: [], 3: ['b']}
    """

    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class CTDataSet:
    """ TODO: optimal dataset backbone """

    def __init__(self, root_dir=str, name: str = 'dataset'):
        self.name = name
        self.root_dir = root_dir
        # NOTE: map everything to index
        self.ct_dict = {}
        # NOTE: {index: [(x, y, z), ...], ...}
        self.centroid_dict = {}

    def __ct_dir_tree__(self):
        """ initialize self.ct_dict """
        raise NotImplementedError

    def _get_position(self, idx: int):
        raise NotImplementedError

    def load_ct_np(self, idx, query_type='index'):
        raise NotImplementedError

    def load_centroid(self, idx, query_type='index'):
        raise NotImplementedError

    def load_seg_np(self, idx, query_type='index'):
        raise NotImplementedError
