from .lidc_tasks import TaskVolume, TaskMalignancy, TaskTexture, TaskSpiculation, TaskSubtlety
from .lndb_tasks import LNDbTaskTexture, LNDbTaskVolume
from .stanfordradiogenomics_tasks import StfRG

TASK_DICT = {'volume': TaskVolume,
             'malignancy': TaskMalignancy,
             'texture': TaskTexture,
             'spiculation': TaskSpiculation,
             'subtlety': TaskSubtlety,
             "LNDbTaskVolume": LNDbTaskVolume,
             "LNDbTaskTexture": LNDbTaskTexture,
             "StfRG": StfRG, }


def get_task(task_name):
    if task_name in TASK_DICT:
        return TASK_DICT[task_name]
    elif task_name.startswith("Stf"):
        return TASK_DICT['StfRG']
    else:
        raise NotImplementedError(f"task name = {task_name}")
    pass
