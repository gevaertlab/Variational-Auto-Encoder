""" 
registering datasets here, 
provide function that can return a list of dataset images, center_point coord and file name
Deprecated
"""
import json
import os
from utils.funcs import print_dict
from utils.io import load_dcm


class Register:

    def __init__(self, name, root_path, info_dict_save_path=None):
        """initialize dataset registers

        Args:
            name ([str]): [name of this register]
            root_path ([str]): [root path of the dataset]
            info_dict_save_path ([str], optional): [save the info dictionary]. Defaults to None.
        """
        self.name = name
        self.root_path = root_path
        self.info_dict_save_path = info_dict_save_path
        # example of info_dict (what should it contain)
        self.info_dict = {'data_dict': {},
                          'root_path': root_path,
                          'name': name,
                          'load_func': self.load_func,  # default loading function is load_dcm
                          'spacing': (1, 1, 1),  # target spacing
                          'transpose_axis': None}  # whether to tranpose image while loading
        pass

    def load_func(self, path):
        return load_dcm(path)

    def register_check(self, scan_list):
        # load saved dict
        if os.path.exists(self.info_dict_save_path):
            self.info_dict = self.load_info_dict()
            # if len matched, then pass
            if len(scan_list) <= len(self.info_dict['data_dict'].keys()):
                print(f"dataset {self.name} already registered")
                return True
        return False

    def load_info_dict(self):
        self.info_dict = self.load_json(self.info_dict_save_path)
        self.info_dict['load_func'] = self.load_func
        return self.info_dict

    def register(self):
        raise NotImplementedError

    def save_info_dict(self, info_dict_save_path=None):
        if not info_dict_save_path:
            if not self.info_dict_save_path:
                raise ValueError("have to specify save path")
        else:
            self.info_dict_save_path = info_dict_save_path
        # don't save function object
        content = self.info_dict.copy()
        content['load_func'] = None
        self.save_json(content, self.info_dict_save_path)

    @staticmethod
    def save_json(content, path):
        with open(path, 'w') as fp:
            json.dump(content, fp)
        pass

    def load_json(self, path=None):
        if not path:
            if not self.info_dict_save_path:
                raise("not path specified")
        else:
            self.info_dict_save_path = path
        with open(self.info_dict_save_path, 'r') as fp:
            content = json.load(fp)
        return content

    def load_func(self):
        raise NotImplementedError

    def __str(self):
        content = self.info_dict.copy()
        # change data_dict and load_func item
        content['data_dict'] = f"{len(content['data_dict'])} patches"
        content['load_func'] = content['load_func'].__name__
        return print_dict(content, title=['name', 'value'])
