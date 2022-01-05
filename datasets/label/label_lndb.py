""" match label for lndb dataset """


from datasets.ct.ct_lndb import LNDbDataset
from .label_ds import Label


class LabelLNDb(Label):

    def __init__(self, dataset=LNDbDataset(), name='volume') -> None:
        super(LabelLNDb, self).__init__(name=name)
        self.dataset = dataset
        self.label_name = name
        pass

    def match_label(self, data_name: str):
        pid, lesion_id = data_name.split('.')
        index = str(int(pid.split('LNDb-')[1]))
        metadata = self.dataset.get_info(index)
        info = metadata[1]['meta_dict'][lesion_id][self.label_name]
        return info


class LabelLNDbVolume(LabelLNDb):

    def __init__(self, dataset=LNDbDataset(), name='volume') -> None:
        super().__init__(dataset=dataset, name=name)


class LabelLNDbTexture(LabelLNDb):

    def __init__(self, dataset=LNDbDataset(), name='texture') -> None:
        super().__init__(dataset=dataset, name=name)
