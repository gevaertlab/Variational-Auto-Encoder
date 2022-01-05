from datasets.ct.ct_stanfordradiogenomics import StanfordRadiogenomicsDataset
from .label_ds import Label


class LabelStanfordRadiogenomics(Label):

    def __init__(self, dataset=StanfordRadiogenomicsDataset(),
                 name: str = 'Tumor',
                 **kwds):
        super().__init__(name=name, **kwds)
        self.dataset = dataset
        self.label_name = name
        pass

    def match_label(self, data_name: str):
        pid = data_name.split('.')[0]
        metadata = self.dataset.get_info(pid)
        info = metadata[1]['meta_info'][self.label_name]
        return info


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
