from .tasks import TaskVolume, TaskMalignancy, TaskTexture, TaskSpiculation, TaskSubtlety
from .labels import LabelVolume, LabelMalignancy, LabelTexture, LabelSpiculation, LabelSubtlety

TASK_DICT = {'volume': TaskVolume,
             'malignancy': TaskMalignancy,
             'texture': TaskTexture,
             'spiculation': TaskSpiculation,
             'subtlety': TaskSubtlety}

LABEL_DICT = {'volume': LabelVolume,
              'malignancy': LabelMalignancy,
              'texture': LabelTexture,
              'spiculation': LabelSpiculation,
              'subtlety': LabelSubtlety}
