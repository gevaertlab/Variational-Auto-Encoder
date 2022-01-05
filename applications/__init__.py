from .tasks.lidc_tasks import TaskVolume, TaskMalignancy, TaskTexture, TaskSpiculation, TaskSubtlety
from .tasks.lndb_tasks import LNDbTaskTexture, LNDbTaskVolume
from .tasks.stanfordradiogenomics_tasks import StfRG

TASK_DICT = {'volume': TaskVolume,
             'malignancy': TaskMalignancy,
             'texture': TaskTexture,
             'spiculation': TaskSpiculation,
             'subtlety': TaskSubtlety,
             "LNDbTaskVolume": LNDbTaskVolume,
             "LNDbTaskTexture": LNDbTaskTexture,
             "StfRG": StfRG}
