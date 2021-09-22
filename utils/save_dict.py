""" defines a dict that contains a saving module of json/npy (a dictionary that saves itself at real time) """


from collections import UserDict
import json
from typing import Optional, Union
import os
import numpy as np


class JsonDict(UserDict):

    def __init__(self, save_path, _dict: Union[dict, None] = {}, **kwargs) -> None:
        super(JsonDict, self).__init__(__dict=_dict, **kwargs)
        self.save_path = save_path
        if os.path.exists(save_path):
            try:
                data = self.load()
            except Exception:
                pass
            self.data.update(data)
        pass

    def save(self):
        with open(self.save_path, "w") as jf:
            json.dump(self.data, jf)
        pass

    def load(self):
        with open(self.save_path, "r") as jf:
            content = json.load(jf)
        self.data.update(content)
        return content


class NpyDict(JsonDict):

    def __init__(self, save_path, _dict: Union[dict, None] = {}, **kwargs):
        super(NpyDict, self).__init__(save_path=save_path,
                                      _dict=_dict,
                                      **kwargs)
        pass

    def save(self):
        np.save(self.save_path, self.data)
        pass

    def load(self):
        self.data = np.load(self.save_path, allow_pickle=True).item()
        return self.data
