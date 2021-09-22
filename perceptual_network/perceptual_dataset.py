import os

import SimpleITK as sitk
from applications.labels import Label
from datasets import REGISTERED_DATASETS
from applications import LABEL_DICT


# TODO: implement dataset
def set_dataset(registered_ds_name: str, label_name: str):
    ds = REGISTERED_DATASETS[registered_ds_name]
    label = LABEL_DICT[label_name]
    
    pass
