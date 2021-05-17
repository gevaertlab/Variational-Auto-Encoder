from .tasks import TaskVolume, TaskMalignancy, TaskTexture, TaskSpiculation, TaskSubtlety

TASK_DICT = {'task_volume': TaskVolume,
             'task_malignancy': TaskMalignancy,
             'task_texture': TaskTexture,
             'task_spiculation': TaskSpiculation,
             'task_subtlety': TaskSubtlety}
