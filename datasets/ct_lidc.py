""" CT dataset for LIDC, largely replaced by pylidc module """

from typing import Union
import numpy as np
from .ct_ds import CTDataSet
import os
import re
from time import time
import xml.etree.ElementTree as ET

import SimpleITK as sitk
import pandas as pd
import pydicom
import pylidc as pl
from configs.config_vars import DS_ROOT_DIR


class LIDCDataSet(CTDataSet):
    '''
    lidc = LIDC()
    header = lidc.loadCT('LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192', 
                         only_header=True)
    '''

    def __init__(self, root_dir: str = None, name='LIDC'):
        if root_dir is None:
            root_dir = os.path.join(DS_ROOT_DIR, "TCIA_LIDC")
        super().__init__(root_dir=root_dir, name=name)
        self.__ct_dir_tree__()
        self.scans = pl.query(pl.Scan)
        pass

    def __ct_dir_tree__(self):
        """ initialize self.ct_dict """
        self.ct_dict['index'] = {}
        self.ct_dict['file_path'] = {}
        self.ct_dict['patient_id'] = {}
        for i in range(self.scans.count()):
            self.ct_dict['index'][i] = i
            self.ct_dict['file_path'][self.scans[i].get_path_to_dicom_files()] = i
            self.ct_dict['patient_id'][self.scans[i].patient_id] = i
            self.centroid_dict[i] = self.get_centroid_lst(self.scans[i])
        return

    def get_centroid_lst(self, scan):
        annos = sum(scan.cluster_annotations(), [])
        centroids = [anno.centroid for anno in annos]
        # round and convert to tuple
        centroids = [tuple(np.rint(cen).astype(int)) for cen in centroids]
        return centroids

    def load_ct_np(self, idx, query_type='index'):
        """ integer for index, string for file_path or patient_id """
        idx = self.ct_dict[query_type][idx]
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == idx)[0]
        return scan.to_volume()  # TODO: test shape and spacing!!!

    def load_centroid(self, idx, query_type='index'):
        idx = self.ct_dict[query_type][idx]
        return self.centroid_dict[idx]

    def get_patient_id(self, file_path):
        '''
        Get scan id as in "nodule_list.csv" the first 
        string of numbers in the third part of rel_path
        '''
        return file_path.split('/')[-3]

    def __len__(self):
        return len(self.ct_dict['index'].keys())
