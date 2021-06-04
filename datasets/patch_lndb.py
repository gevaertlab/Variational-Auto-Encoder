from .patch_ds import PatchDataset


class LNDbPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch/'
        super(LNDbPatchDataset, self).__init__(*args, **kwargs)
