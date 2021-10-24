
from typing import Tuple, Union
from .python_logger import get_logger
from time import time


class Timer:

    def __init__(self, name=Union[Tuple, None]):
        """
        name = (module_name, class_name)
        """
        self.tik = 0
        self.tok = 0
        self.counting = False
        if name:
            module_name, class_name = name
            self.logger = get_logger(class_name)
        else:
            self.logger = get_logger(self.__class__.__name__)
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
