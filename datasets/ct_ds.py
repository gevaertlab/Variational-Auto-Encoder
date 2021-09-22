""" 
CT dataset
The CT dataset that can has several functionalities:
    1. load CT <numpy images of HU units>;
    2. position of nodule <centroid>;
    3. name;
"""


import functools
import json
import os

from numpy.lib.arraysetops import isin
from .utils import train_val_test_split
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
        self.spacing = spacing
        self.transpose_axis = transpose_axis
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
    def get_info(idx: any):
        raise NotImplementedError

    @get_info.register(str)
    def _(self, idx: str):
        return idx, self.data_dict[idx]

    @get_info.register(int)
    def _(self, idx: int):
        key = list(self.data_dict.keys())[idx]
        return key, self.data_dict[key]

    def load_cached(self):
        info_dict = self.load_json(self.save_path)
        self.basic_init(name=info_dict.get('name', None),
                        root_dir=info_dict.get('root_dir', None),
                        spacing=info_dict.get('spacing', [1, 1, 1]),
                        transpose_axis=info_dict.get('transpose_axis', [0, 1, 2]))
        self.data_dict.update(info_dict['data_dict'])

    def load_json(self, path=None):
        if not path:
            if not self.info_dict_save_path:
                raise("not path specified")
        else:
            self.info_dict_save_path = path
        with open(self.info_dict_save_path, 'r') as fp:
            content = json.load(fp)
        return content

    def _get_content_dict(self):
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


class CTDataset:
    """ optimal dataset backbone """
    SPLIT_SET = {'all', 'train', 'val', 'test'}

    def __init__(self,
                 root_dir: str,
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
        pass

    def __len__(self):
        return len(self._ds_info.data_dict)

    def update_info(self, name, info_dict):
        self._ds_info.update_info(name, info_dict)
        pass

    def set_split(self,
                  split='all',
                  ratio=0.1):

        assert split in self.SPLIT_SET, "split invalid"
        if split == 'all':
            self._ds_info.data_dict
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
    
    def generate_ds_params(self):
        return self._ds_info._get_content_dict()
    
    def __str__(self):
        return self._ds_info.__str__()

    def load_info(self, key):
        return getattr(self._ds_info, key)
    
    def _set_ds_info(self):
        raise NotImplementedError
    
    def load_ct_np(self, idx, query_type='index'):
        raise NotImplementedError

    def load_seg_np(self, idx, query_type='index'):
        raise NotImplementedError
