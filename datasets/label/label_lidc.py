""" match label for lidc dataset """

from datasets.ct.ct_lndb import LNDbDataset
from .label_ds import Label
import pylidc as dc


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


class LabelLidc(Label):

    def get_scan_ann_from_file_name(self, data_name: str):
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        patient_id, ann_id = data_name.split('.')[0], data_name.split('.')[1]
        scan = dc.query(dc.Scan).filter(
            dc.Scan.patient_id == patient_id).first()
        ann = dc.query(dc.Annotation).filter(
            dc.Annotation.id == ann_id).first()
        return scan, ann


class LabelVolume(LabelLidc):

    def __init__(self,
                 name: str = 'volume'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.volume
        return value


class LabelMalignancy(LabelLidc):

    def __init__(self,
                 name: str = 'malignancy'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> int
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.malignancy
        return value


class LabelTexture(LabelLidc):

    def __init__(self,
                 name: str = 'texture'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.texture
        return value


class LabelSpiculation(LabelLidc):

    def __init__(self,
                 name: str = 'spiculation'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.spiculation
        return value


class LabelSubtlety(LabelLidc):

    def __init__(self,
                 name: str = 'subtlety'):
        super().__init__(name=name)

    def match_label(self, data_name: str):  # -> float
        """
        :param: data_name: ([patient_id].[ann_id]).nrrd
        """
        scan, ann = self.get_scan_ann_from_file_name(data_name)
        value = ann.subtlety
        return value
