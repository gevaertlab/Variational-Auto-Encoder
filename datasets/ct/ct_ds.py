""" 
CT dataset
The CT dataset that can has several functionalities:
    1. load CT <numpy images of HU units>;
    2. position of nodule <centroid>;
    3. name;
"""


from typing import Any
from datasets.utils import LRUCache
import functools
import json
import os

from numpy.lib.arraysetops import isin
from torch.utils.data.dataset import Dataset
from datasets.utils import train_val_test_split
from utils.funcs import print_dict


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
            print(f"[CTInfoDict] No cached data loaded at: \
                  \'{self.save_path}\'")
        pass

    def update_info(self,
                    name: str,
                    info_dict: dict):
        if name in self.data_dict:
            self.data_dict.update(info_dict)
        else:
            self.data_dict[name] = info_dict
        pass

    @functools.singledispatch
    def get_info(self, idx: Any):
        raise NotImplementedError

    @get_info.register(str)
    def _(self, idx: str):
        return idx, self.data_dict[idx]

    @get_info.register(int)
    def _(self, idx: int):
        key = list(self.data_dict.keys())[idx]
        return key, self.data_dict[key]

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
                 name: str = 'dataset',
                 params={}):
        self.name = name
        self.root_dir = root_dir
        self._split = None
        self._ds_info = CTInfoDict(name=name,
                                   root_dir=root_dir,
                                   **params)
        if not self._ds_info.data_dict:
            self.register()
        self.load_funcs = {}
        pass

    def __len__(self):
        return len(self._ds_info.data_dict)

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
        print(f"[CTDataset] Registering dataset: {self.name}")
        self._set_ds_info()
        self._ds_info.save_cache()
        pass

    def _get_ct_path(self, idx, query_type='index'):
        # get ct path from input idx
        raise NotImplementedError

    def _get_seg_path(self, idx, query_type='index'):
        # get seg path from input idx
        raise NotImplementedError

    # def generate_ds_params(self):
    #     """
    #     generate the information for extraction, key function for preprocessing
    #     e.g. {'...':..., ..., data_dict: {'LIDC-IDRI-0078.1': [...], ...}}
    #     TODO: will be deprecated
    #     """
    #     content_dict = self._ds_info._get_content_dict()
    #     content_dict['load_func'] = self.load_ct_np
    #     data_dict = {}  # reformat the data_dict
    #     for key, value in content_dict['data_dict'].items():
    #         for nodule_id, centroid in value['centroid_dict'].items():
    #             name = '.'.join([key, nodule_id])
    #             path = value['path']
    #             data_dict[name] = [path, centroid]
    #     content_dict['data_dict'] = data_dict
    #     return content_dict

    def __str__(self):
        return self._ds_info.__str__()

    def get_info(self, key):
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
        ct_path = self._get_ct_path(idx, query_type=query_type)
        return self.load_funcs['ct'](ct_path)

    def load_seg_np(self, idx, query_type='index'):
        seg_path = self._get_ct_path(idx, query_type=query_type)
        return self.load_funcs['seg'](seg_path)

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
