from .tasks import LNDbTaskTexture, LNDbTaskVolume, TaskVolume, TaskMalignancy, TaskTexture, TaskSpiculation, TaskSubtlety

TASK_DICT = {'volume': TaskVolume,
             'malignancy': TaskMalignancy,
             'texture': TaskTexture,
             'spiculation': TaskSpiculation,
             'subtlety': TaskSubtlety,
             "LNDbTaskVolume": LNDbTaskVolume,
             "LNDbTaskTexture": LNDbTaskTexture}
