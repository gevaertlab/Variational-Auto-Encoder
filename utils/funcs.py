''' Util functions '''

import os
import yaml
from time import time


def getVersion(path):
    ''' Get current version by increment the previous version '''
    folder_lst = os.listdir(path)
    return max([int(folder.split('_')[1]) for folder in folder_lst]) + 1


def saveConfig(path, config_file):
    with open(os.path.join(path,
                           'config.yaml'),
              'w') as f:
        yaml.dump(config_file, f)


def mkdir_safe(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)
    pass


class Timer:

    def __init__(self):
        self.tik = 0
        self.tok = 0
        self.counting = False
        pass

    def start(self):
        self.counting = True
        self.tik = time()
        pass

    def end(self):
        self.tok = time()
        pass

    def show(self, name=""):
        name = name + ' '
        time_used = self.tok - self.tik
        print(f"{name}| {round(time_used, 0)} secs.")
        pass

    def renew(self):
        self.tik = 0
        self.tok = 0
        self.counting = False

    def __call__(self, name=''):
        if not self.counting:
            self.start()
        else:
            self.end()
            self.show(name)
            self.renew()
        pass


def get_order(lst, ref_lst):
    """ 
    get order of list according to ref_lst 
    e.g. 1
    lst = ['a', 'c', 'b', 'd']
    ref_lst = ['a', 'b', 'c', 'd']
    return = [0, 2, 1, 3]

    e.g. 2
    lst = ['a', 'c', 'd']
    ref_lst = ['a', 'b', 'c', 'd']
    return = [0, 2, 3]

    e.g. 3
    lst = ['a', 'c', 'b', 'd']
    ref_lst = ['b', 'c', 'd']
    return = [0, 2, 1, 3] # don't need to give order for 
    """
    ref_order = {ref: i for (i, ref) in enumerate(ref_lst)}
    # NOTE: give -1 to the unknown order elements,
    # and in reorder function, those will be excluded
    return [ref_order[ele] if ele in ref_order else -1 for ele in lst]


def reorder(lst, order):
    """ reorder the array according to the order """
    # NOTE: exclude elements with order = -1
    return [ele for o, ele in sorted(zip(order, lst)) if o >= 0]
