""" CT Dataset for LIDC """
import logging
import os
import os.path as osp
import re
from typing import Dict, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from configs.config_vars import DS_ROOT_DIR
from utils.io import load_mhd
from utils.python_logger import get_logger

from .ct_ds import CTDataset


class LNDbDataset(CTDataset):
    """
    ct = lndb.load_ct_np(idx)
    """

    def __init__(self,
                 root_dir: str,
                 name: str = 'dataset',
                 split='train',
                 params=...):

        if root_dir is None:
            root_dir = osp.join(DS_ROOT_DIR, "LNDb")

        super().__init__(root_dir,
                         name=name,
                         params=params)
        self.load_funcs['ct'] = load_mhd
        self.impl_set = ['all', 'train', 'test']
        self.set_split(split)
        self.logger = get_logger((__file__, self.__class__.__name__))
        pass

    def _set_ds_info(self):
        """
        initializing self._ds_info
        """
        if not self._split:
            self.logger.info("split not set, default to be \"all\".")
        self.set_split('all')
        filelist = self._get_files()
        meta_csv = self._get_meta_csv()
        # pid = LNDbID
        # path = path
        # centroid_dict {'findingid':FindingID, 'centroid':[x, y, z], 'volume':[va, vb, ..]}
        # rather slow process but small dataset
        for filepath in filelist:
            pid = int('LNDb-(\d+).mhd', re.search(os.path.basename(filepath)))
            idmeta = meta_csv[meta_csv['LNDbID'] == pid]
            centroid_dict = {}
            findingids = idmeta['FindingID'].unique()
            # NOTE: didn't add segmentation yet masks: LNDb-{pid 4 digit}_rad{radid}.mhd
            for findingid in findingids:
                centroid_dict[findingid] = {}
                fidmeta = idmeta[idmeta['FindingID'] == findingid]
                # extract coord by taking the mean
                centroid_dict[findingid]['centroid'] = fidmeta[[
                    'x', 'y', 'z']].mean()
                # TODO: postprocessing
                # add volume
                centroid_dict[findingid]['volumes'] = fidmeta['Volume'].tolist()
                # add texture
                centroid_dict[findingid]['texture'] = fidmeta['Text'].mean()
            self.update_info(pid, {'pid': pid,
                                   'path': filepath,
                                   'centroid_dict': centroid_dict})
        pass

    def _valid_split(self, split: str):
        try:
            assert split in self.impl_set
        except AssertionError:
            self.logger.exception(f"split {split} not in {self.impl_set}")
        pass

    def set_split(self, split):
        self._valid_split(split)
        self._split = split
        pass

    def _get_files(self):
        if not self._split:
            self.logger.info("split not set, default to be \"all\".")
        self.set_split('all')
        if self._split == 'train':
            return self._get_file_from_root(osp.join(self.root_dir, 'trainset'))
        elif self._split == 'test':
            return self._get_file_from_root(osp.join(self.root_dir, 'testset'))
        elif self._split == 'all':
            return self._get_file_from_root(osp.join(self.root_dir, 'trainset')) + \
                self._get_file_from_root(osp.join(self.root_dir, 'testset'))
        else:
            self.logger.warning("split {self._split} not implemented")
            return []

    @staticmethod
    def _get_file_from_root(root_path, file_ext):
        filepathlist = []
        for root, dirs, files in os.walk(root_path, topdown=False):
            for file in files:
                if file.endswith(file_ext):
                    filepathlist.append(osp.join(root, file))
        return filepathlist

    def _get_meta_csv(self, path=None):
        """ load metadata csv, hard code """
        if path:
            metacsv = pd.read_csv(path)
        else:
            if self._split == "train":
                path = osp.join(self.root_dir, 'trainset', 'trainNodules.csv')
                metacsv = self._get_meta_csv(path=path)
            elif self._split == 'test':
                path = osp.join(self.root_dir, 'testset', 'testNodules.csv')
                metacsv = self._get_meta_csv(path=path)
            elif self._split == 'all':
                metacsv = pd.concat([self._get_meta_csv(path=osp.join(self.root_dir, 'trainset', 'trainNodules.csv')),
                                     self._get_meta_csv(path=osp.join(self.root_dir, 'testset', 'testNodules.csv'))],
                                    axis=0)
        # postprocessing
        # 1. delete non nodules
        metacsv = metacsv[metacsv['Nodule'] == 1]
        # 2. keep one LNDbID + FindingID combination
        return metacsv

    def _get_ct_path(self, idx):
        return self.get_info(idx)['path']

    def load_ct_np(self, idx, query_type='index'):
        """ integer for index, string for file_path or patient_id """
        return super().load_ct_np(idx)

    def load_centroid(self, idx, query_type='index'):
        return self.get_info(idx)['centroid_dict']['centroid']


class LNDBDataset(CTDataset):
    '''
    lndb = LNDB()
    ct = lndb.loadCT(306)
    '''

    def __init__(self,
                 root_dir: str = None,
                 name='LIDC',
                 params={},
                 reset_info=False):
        if root_dir is None:
            root_dir = osp.join(DS_ROOT_DIR, 'LNDb/trainset')
        super().__init__(root_dir=root_dir, name=name)
        # TODO: read csv; specific for LNDb, need metadata file for centroid information
        # for every new dataset, need function handle meta_file that loads metadata in a standardized way.

        self.params = self.handle_params(params)
        self.__ct_dir_tree__()
        self.path_dict = {}
        pass

    def handle_params(params: Dict):
        # handles meta data
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
