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
