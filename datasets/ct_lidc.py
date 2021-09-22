""" CT dataset for LIDC, largely replaced by pylidc module """

from typing import Union
import numpy as np
from tqdm import tqdm
from .ct_ds import CTDataset
import os
import os.path as osp
import re
from time import time
import xml.etree.ElementTree as ET

import SimpleITK as sitk
import pandas as pd
import pydicom
import pylidc as pl
from configs.config_vars import DS_ROOT_DIR
from utils.io import load_dcm


class LIDCDataSet(CTDataset):
    '''
    lidc = LIDC()
    header = lidc.loadCT('LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192', 
                         only_header=True)
    '''

    def __init__(self,
                 root_dir: str = None,
                 name='LIDC',
                 params={},
                 reset_info=False):
        if root_dir is None:
            root_dir = os.path.join(DS_ROOT_DIR, "TCIA_LIDC")
        super().__init__(root_dir=root_dir, name=name, params=params)
        if not self._ds_info.data_dict or reset_info:
            self._set_ds_info()
        pass

    def _set_ds_info(self):
        """ initialize self._ds_info """
        scans = pl.query(pl.Scan)
        for i in tqdm(range(scans.count())):
            pid = scans[i].patient_id
            path = scans[i].get_path_to_dicom_files()
            centroid_dict = self.get_centroid_dict(scans[i])
            self.update_info(pid, {'pid': pid,
                                   'path': path,
                                   'index': i,
                                   'centroid_dict': centroid_dict})
        pass

    def get_centroid_dict(self, scan):
        """ get all the centroid points from a scan """
        nodules = scan.annotations
        centroids = {}
        for nodule in nodules:
            x, y, z = nodule.centroid
            centroid = (x, y, z)
            centroids[nodule.id] = centroid
        return centroids

    def load_ct_np(self, idx):
        """ integer for index, string for file_path or patient_id """
        if isinstance(idx, str) and osp.exists(idx):
            path = idx
        else:
            info = self._ds_info.get_info(idx)
            path = info['path']
        # scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == idx)[0]
        # data = scan.to_volume()
        return load_dcm(path)

    def load_centroid(self, idx, query_type='index'):
        idx = self.ct_dict[query_type][idx]
        return self.centroid_dict[idx]

    def get_patient_id(self, file_path):
        '''
        Get scan id as in "nodule_list.csv" the first 
        string of numbers in the third part of rel_path
        '''
        return file_path.split('/')[-3]

    def generate_ds_params(self):
        """ 
        very specifically focused function, generate the information for extraction.
        e.g. {'...':..., ..., data_dict:\{'LIDC-IDRI-0078.1': [...], ...\}}
        """
        content_dict = self._ds_info._get_content_dict()
        content_dict['load_func'] = self.load_ct_np
        data_dict = {} # reformat the data_dict
        for key, value in content_dict['data_dict'].items():
            for nodule_id, centroid in value['centroid_dict'].items():
                name = '.'.join([key, nodule_id])
                path = value['path']
                data_dict[name] = [path, centroid]
        content_dict['data_dict'] = data_dict
        return content_dict
