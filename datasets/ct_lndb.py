""" CT dataset for LIDC """
from typing import Union
import numpy as np
from .ct_ds import CTDataSet
import os
import os.path as osp
import re

import SimpleITK as sitk
from configs.config_vars import DS_ROOT_DIR


class LNDBDataSet(CTDataSet):
    '''
    lndb = LNDB()
    ct = lndb.loadCT(306)
    '''

    def __init__(self, root_dir: str = None, name='LNDB_train'):
        if root_dir is None:
            root_dir = osp.join(DS_ROOT_DIR, 'LNDb/trainset')
        super().__init__(root_dir=root_dir, name=name)
        self.__ct_dir_tree__()
        self.path_dict = {}
        pass

    def __ct_dir_tree__(self):
        """ initialize self.ct_dict """
        self.ct_dict['index'] = {}
        self.ct_dict['file_path'] = {}
        self.ct_dict['file_name'] = {}
        file_names, file_paths = self._get_lndb_files()
        for (file_name, file_path) in zip(file_names, file_paths):
            idx = self.get_id(file_name)
            self.path_dict[idx] = file_path.split('.')[0]
            self.ct_dict['index'][idx] = idx
            self.ct_dict['file_path'][file_path] = idx
            self.ct_dict['file_name'][file_name] = idx
            self.centroid_dict[idx] = []
        return

    def get_id(self, file_name):  # -> int
        # HACK: hard code
        p = re.compile("LNDb-([0-9]+).[mhd|raw]")
        result = p.search(file_name)
        return int(result.group(1))

    def _idx2str(self, idx: int):
        return str(idx).zfill(4)

    def _get_lndb_files(self):
        folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(f)]
        file_names = []
        file_paths = []
        for folder in folders:
            file_name_sub_list = os.listdir(
                os.path.join(self.root_dir, folder))
            file_names += file_name_sub_list
            file_paths += [os.path.join(self.root_dir, folder, f)
                           for f in file_name_sub_list]
        return file_names, file_paths

    def load_ct_np(self, idx, query_type='index'):
        """ integer for index, string for file_path or patient_id """
        idx = self.ct_dict[query_type][idx]
        file_path = self.path_dict[idx]
        ct_sitk = self.load_ct_sitk(self, file_path)
        ct_np = sitk.GetArrayFromImage(ct_sitk)
        # TODO: test shape and spacing!!!
        return ct_np

    def load_ct_sitk(self, file_path):
        return sitk.ReadImage(file_path)

    def load_centroid(self, idx, query_type='index'):
        idx = self.ct_dict[query_type][idx]
        return self.centroid_dict[idx]
