from .task_base import TaskBase


class LNDbTaskVolume(TaskBase):

    def __init__(self,
                 name: str = 'lndb_volmne',
                 task_type: str = 'regression'):
        super().__init__(name=name, task_type=task_type)


class LNDbTaskTexture(TaskBase):

    def __init__(self,
                 name: str = 'lndb_texture',
                 task_type: str = 'classification'):
        super().__init__(name=name, task_type=task_type)
