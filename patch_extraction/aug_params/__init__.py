
# from patch_extraction.aug_params import shift, rotate
from .single import AUG_PARAMS as single_aug_params
from .mixed import AUG_PARAMS as mixed_aug_params

AUG_PARAMS = {"": {}}
AUG_PARAMS.update(single_aug_params)
AUG_PARAMS.update(mixed_aug_params)
