""" 
registering datasets here, 
provide function that can return a list of dataset images, center_point coord and file name
"""
import json


class Register:

    def __init__(self, name, root_path, info_dict_save_path=None):
        self.name = name
        self.root_path = root_path
        self.info_dict_save_path = info_dict_save_path
        self.info_dict = {'data_dict': {},
                          'root_path': root_path,
                          'name': name,
                          'load_func':None}
        pass

    def register(self):
        raise NotImplementedError

    def save_info_dict(self, info_dict_save_path=None):
        if not info_dict_save_path:
            if not self.info_dict_save_path:
                raise ValueError("have to specify save path")
        else:
            self.info_dict_save_path = info_dict_save_path
        self.save_json(self.info_dict, self.info_dict_save_path)

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
