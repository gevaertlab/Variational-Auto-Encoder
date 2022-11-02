""" 
CT dataset
The CT dataset that can has several functionalities:
    1. load CT <numpy images of HU units>;
    2. position of nodule <centroid>;
    3. name;
"""


import json
import os
import os.path as osp
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import SimpleITK as sitk
from datasets.utils import LRUCache, train_val_test_split
from numpy.lib.arraysetops import isin
from torch.utils.data.dataset import Dataset
from utils.funcs import print_dict
from utils.python_logger import get_logger


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


class CT:

    def __init__(self,
                 save_path: str,
                 meta_data=None):
        self.save_path = save_path
        self.meta_data = meta_data
        pass

    def get_meta(self, name):
        return self.meta_data.get(name, None)

    def get_ext(self):
        return self.save_path.split('.')[-1]

    def _exist(self):
        return os.path.exists(self.save_path)


class CTInfoDict:

    def __init__(self,
                 name=None,
                 root_dir=None,
                 spacing=[1, 1, 1],
                 transpose_axis=[0, 1, 2],
                 save_path=None):
        """

        Args:
            name ([type], optional): [description]. Defaults to None.
            root_dir ([type], optional): [description]. Defaults to None.
            spacing (list, optional): [description]. Defaults to [1, 1, 1].
            transpose_axis (list, optional): [description]. Defaults to [0, 1, 2].
            data_dict: (e.g. {<filename>:{'path':<path>, 'centroid':<centroid>}})
        """
        self.logger = get_logger(self.__class__.__name__)
        self.basic_init(name=name,
                        root_dir=root_dir,
                        spacing=spacing,
                        transpose_axis=transpose_axis)
        self.save_path = save_path
        self.data_dict = {}
        self.advance_init()
        pass

    def basic_init(self,
                   name=None,
                   root_dir=None,
                   spacing=[1, 1, 1],
                   transpose_axis=[0, 1, 2]):
        self.name = name
        self.root_dir = root_dir
        self.spacing = spacing  # target spacing, no need to change
        self.transpose_axis = transpose_axis  # target axis, no need to change
        pass

    def advance_init(self):
        if self.save_path is None:
            return
        elif os.path.exists(self.save_path):
            self.load_cached()
        else:
            self.logger.info(f"No cached data loaded at: \'{self.save_path}\'")
        pass

    def update_info(self,
                    name: str,
                    info_dict: dict):
        if name in self.data_dict:
            self.data_dict[name].update(info_dict)
        else:
            self.data_dict[name] = info_dict
        pass

    def get_info(self, idx: Any):
        if idx in self.data_dict.keys():
            return idx, self.data_dict[idx]
        elif isinstance(idx, int):
            key = list(self.data_dict.keys())[idx]
            return key, self.data_dict[key]
        elif osp.isdir(idx) or osp.isfile(idx):
            info = [(key, value)
                    for key, value in self.data_dict.items() if value['path'] == idx]
            return info[0]

    def load_cached(self):
        """load cached info dict"""
        info_dict = self.load_json(self.save_path)
        self.basic_init(name=info_dict.get('name', None),
                        root_dir=info_dict.get('root_dir', None),
                        spacing=info_dict.get('spacing', [1, 1, 1]),
                        transpose_axis=info_dict.get('transpose_axis', [0, 1, 2]))
        self.data_dict.update(info_dict['data_dict'])
        pass

    def load_json(self, path=None):
        if not path:
            if not self.info_dict_save_path:
                raise ValueError("not path specified")
        else:
            self.info_dict_save_path = path
        with open(self.info_dict_save_path, 'r') as fp:
            content = json.load(fp)
        return content

    def _get_content_dict(self):
        # data_dict stores all the image information while
        # other params are for the dataset
        content_dict = {'name': self.name,
                        'root_dir': self.root_dir,
                        'spacing': self.spacing,
                        'transpose_axis': self.transpose_axis,
                        'data_dict': self.data_dict}
        return content_dict

    def save_cache(self):
        content_dict = self._get_content_dict()
        self.save_json(content_dict, self.save_path)
        pass

    def __str__(self):
        content_dict = self._get_content_dict()
        content_copy = content_dict.copy()
        # change data_dict and load_func item
        content_copy['data_dict'] = f"length = {len(content_copy['data_dict'])}"
        return print_dict(content_copy, title=['name', 'value'])

    @staticmethod
    def save_json(content, path):
        with open(path, 'w') as fp:
            json.dump(content, fp)
        pass


class CTDataset(Dataset):
    __metaclass__ = ABCMeta

    """
    optimal dataset backbone 
    key implementation:
    1. (optional) __getitem__: can load CT + metadata can change if needed.
    2. load_ct_np: can load numpy ct
    3. (optional) load_seg: can load segmentation if exist
    4. set_ds_info: set and save metadata
    RECOMENNED: same set of index can be used to load ct as well as it's segmentation 
    if there's any, and loading the metadata
    """
    SPLIT_SET = {'all', 'train', 'val', 'test'}

    def __init__(self,
                 root_dir: str,  # for info dict saving
                 split: str = 'train',
                 name: str = 'dataset',
                 params={}):
        self.name = name
        self.root_dir = root_dir
        self._set_split(split)
        self._ds_info = CTInfoDict(name=name,
                                   root_dir=root_dir,
                                   **params)
        # _ds_info should be manually init in each dataset
        # if not self._ds_info.data_dict:
        #     self.register()
        self.load_funcs = {}
        pass

    def _valid_split(self, split):
        return split in self.SPLIT_SET

    def __len__(self):
        return len(self._ds_info.data_dict)

    def _set_split(self, split):
        self._valid_split(split)
        self._split = split
        pass

    def update_info(self, name, info_dict):
        """
        @param: name: str, a key for information retrival
        @param: info_dict: Dict, dictionary of entries and information to store under this key
        e.g. {'path':<path>, 'centroid':<centroid>}
        this entry will look like this: in data_dict entry in _ds_info
        data_dict: (e.g. {<filename>:{'path':<path>, 'centroid':<centroid>}}
        """
        self._ds_info.update_info(name, info_dict)
        pass

    def set_split(self,
                  split='all',
                  ratio=0.1):
        # should be overwritten
        assert split in self.SPLIT_SET, "split invalid"
        if split == 'all':
            return self._ds_info.data_dict
        else:
            idx = {'train': [], 'val': [], 'test': []}
            idx['train'], \
                idx['val'], \
                idx['test'] = train_val_test_split(self.__len__(),
                                                   ratio=ratio,
                                                   random_state=9001)
            split_data = [self._ds_info.get_info(i) for i in idx[split]]
            split_data_dict = {sd[0]: sd[1] for sd in split_data}
            self._split = split
            return split_data_dict

    def register(self):
        print(f"Registering dataset: {self.name}")
        self._set_ds_info()
        self._ds_info.save_cache()
        pass

    def __str__(self):
        return self._ds_info.__str__()

    def get_info(self, key, query_type='index'):
        if key in self._ds_info.__dict__:
            return getattr(self._ds_info, key)
        else:
            return self._ds_info.get_info(key)

    def _set_ds_info(self):
        """
        REQUIRED: function to set up self._ds_info 
        uses self.update_info function
        """
        raise NotImplementedError

    def load_ct_np(self, idx: Any, query_type='index'):
        """ REQUIRED: to load CT as numpy array, you can define what idx is """
        if osp.isdir(idx) or osp.isfile(idx):
            ct_path = idx
        else:
            ct_path = self.get_info(idx, query_type=query_type)[1]['path']
            if isinstance(ct_path, dict):
                ct_path = ct_path['img_path']
             # TODO: test other datasets
        img = self.load_funcs['ct'](ct_path)
        if isinstance(img, sitk.Image):
            return sitk.GetArrayFromImage(img).transpose((2, 1, 0))
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise NotImplementedError(f"img type {type(img)} not implemented")

    def load_seg_np(self, idx, query_type='index'):
        if osp.isdir(idx) or osp.isfile(idx):
            seg_path = idx
        else:
            seg_path = self.get_info(idx, query_type=query_type)[
                1]['path']['seg_path']
        seg = self.load_funcs['seg'](seg_path)
        if isinstance(seg, sitk.Image):
            return sitk.GetArrayFromImage(seg).transpose((2, 1, 0))
        elif isinstance(seg, np.ndarray):
            return seg
        else:
            raise NotImplementedError(f"seg type {type(seg)} not implemented")

    def __getitem__(self, idx, query_type='index'):
        return self.load_ct_np(idx), self.get_info(idx)

    def load_centroid(self, idx, query_type='index'):
        raise NotImplementedError


class CTCachedDataset(CTDataset):

    def __init__(self,
                 cache_capacity=5,
                 *args,
                 **kwargs):
        # cached loaded CT images to accelerate _getitem_
        super(CTCachedDataset, self).__init__(*args, **kwargs)
        self.cache_capacity = cache_capacity
        self.cache = LRUCache(self.cache_capacity)
        pass

    def __getitem__(self, idx: int, query_type='index'):
        if self.cache.has(idx):
            return self.cache.get(idx)
        else:
            value = self.load_ct_np(idx)
            self.cache.put(idx, value)
            return value
