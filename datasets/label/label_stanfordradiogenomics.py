from datasets.ct.ct_stanfordradiogenomics import StanfordRadiogenomicsDataset
from .label_ds import Label
from typing import List


class LabelStanfordRadiogenomics(Label):

    def __init__(self, dataset=StanfordRadiogenomicsDataset(),
                 name: str = 'Tumor',
                 dataset_name=None,
                 **kwds):
        super().__init__(name=name, dataset_name=dataset_name, **kwds)
        self.dataset = dataset
        self.label_name = name
        pass

    def match_label(self, data_name: str):
        # e.g. data_name = 'R01-001.0.Aug00'
        pid = data_name.split('.')[0]
        metadata = self.dataset.get_info(pid)
        label = metadata[1]['meta_info'][self.label_name]
        return label


class LabelStfRG(LabelStanfordRadiogenomics):

    def __init__(self, name='Gender', **kwds) -> None:
        super().__init__(name=name, **kwds)
        pass

    def list_label_names(self):
        metadata = self.dataset.get_info(0)
        return metadata[1]['meta_info'].keys()

    def change_name(self, name='Tumor'):
        self.label_name = name
        super().__init__(name=name)
        pass


class LabelStfRGGender(LabelStanfordRadiogenomics):

    def __init__(self, name='Gender', **kwds) -> None:
        super().__init__(name=name, **kwds)


class LabelStfRGSmoking(LabelStanfordRadiogenomics):

    def __init__(self, name='Smoking status', **kwds) -> None:
        super().__init__(name=name, **kwds)


class LabelStfRGVolume(LabelStanfordRadiogenomics):

    def __init__(self, name='volume', dataset_name="StfRadiogenomics", **kwds) -> None:
        super().__init__(name=name, dataset_name=dataset_name, **kwds)


# new feature, classes are re-grouped.
class LabelStfReGroup(LabelStanfordRadiogenomics):
    """ because handling NA values, so label should all be string !!! """

    def __init__(self, name, regroup_tuples: List, na_value='NA', **kwds) -> None:
        """AI is creating summary for __init__

        Args:
            name ([type]): [description]
            regroup_tuple (Tuple): {e.g. [(["T1", "T1a"], "T1"), (["T2b"], "T2")]}
        """
        super().__init__(name=name, **kwds)
        self.regroup_tuples = regroup_tuples
        assert any(isinstance(t[1], str) for t in self.regroup_tuples), "regroup type should be string"
        self.regroup_dict = self.init_regroup_dict(regroup_tuples)
        self.na_value = na_value
        pass

    def init_regroup_dict(self, regroup_tuples):
        regroup_dict = {}
        for klist, v in regroup_tuples:
            for k in klist:
                regroup_dict[k] = v

        return regroup_dict

    def regroup(self, label):
        # remove first and end spaces
        label = "".join(label.rstrip().lstrip())
        if label in self.regroup_dict:
            return self.regroup_dict[label]
        else:
            return self.na_value

    def match_label(self, data_name: str):
        pid = data_name.split('.')[0]
        metadata = self.dataset.get_info(pid)
        label = metadata[1]['meta_info'][self.label_name]
        return self.regroup(label)


class LabelStfTStage(LabelStfReGroup):

    def __init__(self,
                 name='Pathological T stage',
                 regroup_tuples=[(["T1a", "T1b"], "T1"),
                                 (["T2a", "T2b", "T3", "T4"], "T2+")],
                 **kwds):
        super().__init__(name=name, regroup_tuples=regroup_tuples, kwds=kwds)
        pass


class LabelStfNStage(LabelStfReGroup):

    def __init__(self,
                 name='Pathological N stage',
                 regroup_tuples=[(["N0"], "N0"),
                                 (["N1", "N2"], "N1+")],
                 **kwds):
        super().__init__(name=name, regroup_tuples=regroup_tuples, kwds=kwds)
        pass


class LabelStfAJCC(LabelStfReGroup):

    def __init__(self,
                 name='AJCC Staging (Version 7)',
                 regroup_tuples=[(["IA", "IB"], "I"),
                                 (["IIA", "IIB", "IIIA", "IIIB", "IV"], "II+")],
                 **kwds):
        super().__init__(name=name, regroup_tuples=regroup_tuples, kwds=kwds)
        pass


class LabelStfHisGrade(LabelStfReGroup):

    def __init__(self,
                 name='Histopathological Grade',
                 regroup_tuples=[(["G1 Well differentiated"], "G1"),
                                 (["G2 Moderately differentiated"], "G2"),
                                 (["G3 Poorly differentiated"], "G3")],
                 **kwds):
        super().__init__(name=name, regroup_tuples=regroup_tuples, kwds=kwds)
        pass


class LabelStfRGLymphInvasion(LabelStfReGroup):

    def __init__(self,
                 name='Lymphovascular invasion',
                 dataset_name="StfRadiogenomics",
                 regroup_tuples=[(["Absent"], "Absent"),
                                 (["Present"], "Present")],
                 **kwds) -> None:
        super().__init__(name=name,
                         dataset_name=dataset_name,
                         regroup_tuples=regroup_tuples,
                         **kwds)


class LabelStfRGPleuralInvasion(LabelStanfordRadiogenomics):

    def __init__(self,
                 name='Pleural invasion (elastic, visceral, or parietal)',
                 dataset_name="StfRadiogenomics",
                 **kwds) -> None:
        super().__init__(name=name, dataset_name=dataset_name, **kwds)


class LabelStfEGFRMutation(LabelStfReGroup):

    def __init__(self,
                 name='EGFR mutation status',
                 regroup_tuples=[(["Mutant"], "Mutant"),
                                 (["Wildtype"], "Wildtype"), ],
                 **kwds):
        super().__init__(name=name, regroup_tuples=regroup_tuples, kwds=kwds)
        pass


class LabelStfKRASMutation(LabelStfReGroup):

    def __init__(self,
                 name='KRAS mutation status',
                 regroup_tuples=[(["Mutant"], "Mutant"),
                                 (["Wildtype"], "Wildtype"), ],
                 **kwds):
        super().__init__(name=name, regroup_tuples=regroup_tuples, kwds=kwds)
        pass


# 'Case ID', 'Event Name', 'R01 RNASeq ID', 'U01 RNASeq ID', 'GSM ID', 'Research Study',
# 'Tissue Sample ID', 'Date of tissue sample collection (Identifier)',
# 'Patient Name (Identifier)', 'Anonymized Patient Name', 'Medical Record Number (Identifier)',
# 'Anonymized Medical Record Number', 'Patient affiliation', 'Other Case Notes',
# 'Date of Birth (Identifier)', 'Age at Histological Diagnosis', 'Weight (lbs)',
# 'Gender', 'Ethnicity', 'Smoking status', 'Pack Years', 'Quit Smoking Year',
# 'Primary Tumor', 'Initial Segmentation', 'DICOM on ePAD', 'Date Annotated', '%GG',
# 'Annotation Comments', 'Tumor Location (choice=RUL)', 'Tumor Location (choice=RML)',
# 'Tumor Location (choice=RLL)', 'Tumor Location (choice=LUL)', 'Tumor Location (choice=LLL)',
# 'Tumor Location (choice=L Lingula)', 'Tumor Location (choice=Unknown)', 'Histology ',
# "Histology, if 'Other' selected above", 'Pathological T stage', 'Pathological N stage',
# 'Pathological M stage', 'AJCC Staging (Version 7)', 'Histopathological Grade',
# 'Lymphovascular invasion', 'Pleural invasion (elastic, visceral, or parietal)',
# 'Tumor', 'Tumor total (mg)', 'ADJ', 'ADJ Total (mg)', 'Normal', 'Normal total (mg)',
# 'Blood', 'In EGFR/KRAS cohort', 'EGFR mutation status', 'EGFR mutation details',
# 'KRAS mutation status', 'KRAS mutation detailas', 'ALK translocation status',
# 'Adjuvant Treatment', 'Chemotherapy', 'Radiation', 'FEV1 (L)', 'FEV1 (%)', 'Recurrence',
# 'Recurrence Location', 'Location of Recurrence', 'Date of Recurrence', 'Recurrence Notes',
# 'Date of Last Known Alive', 'Time to Last Follow-up', 'Survival Status', 'Date of Death',
# 'Time to Death (days)', 'Microarray ID', 'Submicroarray ID', 'Tumor RNASeq_ID',
# 'Concentration (ng/ul)', 'RIN (from Centrillion)', 'RNA-seq ', 'CT Accession ',
# 'CT Date', 'Days between CT and surgery', 'Outside CT', 'CT Slice Thickness (mm)',
# 'Iodine Use', 'PET Date', 'CT Slice Number', 'PET Accession', 'Outside PET',
# 'PET Comments', 'PET Slice Number', 'PET/CT slice notes ', 'PET SUVmax 2012',
# 'PET SUVmax ', 'PET SULmax', 'PET SUVpeak', 'PET SULpeak', 'MTV', 'TLG', 'Imaging Notes',
# 'Reason for Not Measured', 'Bi-dimensize by CT', 'Volume by CT', 'Blood Chemistry Date',
# 'WBC', 'Hemoglobin', 'Hematocrit', 'Platelets', 'Complete?', 'Sub-population',
# 'Number of cells', 'RNA-seq batch prefix', 'RNA-seq fastq prefix', 'Number of reads',
# 'Complete?.1'
