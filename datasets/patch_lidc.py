from .patch_ds import PatchDataset


class LIDCPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs): # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-patch/'
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        pass


class LIDCPatch32Dataset(PatchDataset):
    
    def __init__(self, *args, **kwargs): # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-patch-32/'
        super(LIDCPatch32Dataset, self).__init__(*args, **kwargs)
        pass