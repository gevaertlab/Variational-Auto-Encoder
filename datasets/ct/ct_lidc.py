""" CT Dataset for LIDC, largely replaced by pylidc module """


from tqdm import tqdm
from .ct_ds import CTCachedDataset, CTDataset
import os
import os.path as osp

import pylidc as pl
from configs.config_vars import DS_ROOT_DIR
from utils.io import load_dcm


class LIDCDataset(CTDataset):
    '''
    lidc = LIDC()
    header = lidc.load_ct_np('LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192', 
                         only_header=True)
    '''

    def __init__(self,
                 root_dir: str = None,
                 name='LIDC',
                 params={
                     "save_path": "/labs/gevaertlab/data/lungcancer/TCIA_LIDC/lidc_info_dict.json"},
                 reset_info=False):
        if root_dir is None:
            root_dir = os.path.join(DS_ROOT_DIR, "TCIA_LIDC")
        super().__init__(root_dir=root_dir,
                         name=name,
                         params=params)
        self._scans = pl.query(pl.Scan)
        if (not self._ds_info.data_dict) or reset_info:
            self._set_ds_info()  # not saving this
        self.load_funcs['ct'] = load_dcm
        pass

    def _set_ds_info(self):
        """ 
        initialize self._ds_info 
        NOTE: more or less standardized way to do it, need get_centroid_dict that takes
        arbitary argument and can return centroids = {nodule.id: [x, y, z], ...}
        """
        for i in tqdm(range(self._scans.count())):
            pid = self._scans[i].patient_id
            path = self._scans[i].get_path_to_dicom_files()
            centroid_dict = self.load_centroid(i)
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

    def _get_ct_path(self, idx, query_type='index'):
        if isinstance(idx, str) and osp.exists(idx):
            path = idx
        else:
            info = self._ds_info.get_info(idx)
            path = info['path']
        return path

    def load_ct_np(self, idx):
        """ integer for index, string for file_path or patient_id """
        return super().load_ct_np(idx)

    def load_centroid(self, idx, query_type='index'):
        try:
            return self._ds_info.get_info(idx)['centroid']
        except IndexError or KeyError:
            scan = self._scans[idx]
            return self.get_centroid_dict(scan)

    def get_patient_id(self, file_path):
        '''
        Get scan id as in "nodule_list.csv" the first 
        string of numbers in the third part of rel_path
        '''
        return file_path.split('/')[-3]


class LIDCCachedDataset(CTCachedDataset):
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

        return load_dcm(path)

    def load_centroid(self, idx, query_type='index'):
        return self.get_info(idx)['centroid_dict']  # TODO: test it

    def get_patient_id(self, file_path):
        '''
        Get scan id as in "nodule_list.csv" the first 
        string of numbers in the third part of rel_path
        '''
        return file_path.split('/')[-3]
