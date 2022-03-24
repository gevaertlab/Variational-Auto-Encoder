""" CT Dataset for NSCLC """
import os
import os.path as osp

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from utils.io import load_img
from utils.python_logger import get_logger

from .ct_ds import CTDataset


class StanfordRadiogenomicsDataset(CTDataset):

    def __init__(self,
                 root_dir: str = None,
                 name: str = 'StanfordRadiogenomics',
                 split='train',
                 params={
                     "save_path": "/labs/gevaertlab/data/lung cancer/StanfordRadiogenomics/stf_info_dict.json"},
                 reset_info=False):

        if root_dir is None:
            root_dir = "/labs/gevaertlab/data/NSCLC_Radiogenomics/Resampled"  # HACK: hard code this
        self.logger = get_logger(self.__class__.__name__)
        self.impl_set = ['all', 'train', 'test']
        super().__init__(root_dir,
                         split=split,
                         name=name,
                         params=params)

        self.load_funcs['ct'] = load_img
        self.load_funcs['seg'] = load_img
        if (not self._ds_info.data_dict) or reset_info:
            self.register()  # saving this
        pass

    def _get_files(self):
        img_dir = osp.join(self.root_dir, "Images")
        seg_dir = osp.join(self.root_dir, "Segmentations")
        img_files = os.listdir(img_dir)
        seg_dirs = os.listdir(seg_dir)


        # def rep_img(img_file_name):
        #     return img_file_name.replace('img.nii.gz', 'msk.nii.gz')

        # filelist = [(imgf, rep_img(imgf))
        #             for imgf in img_files if rep_img(imgf) in seg_files]

        # NOTE: segmentation files changed

        filelist = []
        for imgf in img_files:
            seg_subdir = imgf.split("_")[0]
            if seg_subdir in seg_dirs:
                segfname = os.listdir(osp.join(seg_dir, seg_subdir))[0]
                segf = osp.join(seg_subdir, segfname)
                filelist.append((imgf, segf))

        return filelist, (img_dir, seg_dir)

    def _get_meta_csv(self):
        return pd.read_csv(osp.join(self.root_dir, "NSCLCR01Radiogenomic_DATA_LABELS.csv"))

    def _extract_meta(self, pid, seg_path, meta_csv):
        meta_df = meta_csv[meta_csv['Case ID'] == pid.replace('-', '')]
        meta_dict = meta_df.to_dict(orient='records')[0]
        seg_np = self.load_seg_np(seg_path)
        centroid = self._get_centroid_from_seg(seg_np)
        volume = self._get_volume_from_seg(seg_np)
        meta_dict.update({"volume": volume})
        return {"pid": pid, "centroid_dict": centroid, "meta_info": meta_dict}

    def _get_volume_from_seg(seg_np):
        seg_binary = (seg_np != 0).astype("uint8")
        return np.sum(seg_binary)

    @staticmethod
    def _get_centroid_from_seg(seg_np):
        # 3D
        seg_bool = (seg_np > 0).astype('int')
        y = np.median(np.nonzero(
            (np.sum(seg_bool, axis=(1, 2)) > 0).astype('int')))
        x = np.median(np.nonzero(
            (np.sum(seg_bool, axis=(0, 2)) > 0).astype('int')))
        z = np.median(np.nonzero(
            (np.sum(seg_bool, axis=(0, 1)) > 0).astype('int')))

        return {0: (x, y, z)}

    def load_seg_np(self, seg_path):
        seg_itk = self.load_seg_itk(str(seg_path))
        return sitk.GetArrayFromImage(seg_itk).transpose()

    def load_seg_itk(self, seg_path):
        return self.load_funcs['seg'](seg_path)

    def _set_ds_info(self):
        filelist, (img_dir, seg_dir) = self._get_files()
        meta_csv = self._get_meta_csv()
        for img, seg in tqdm(filelist):
            pid = img.replace("_img.nii.gz", "")
            meta_dict = self._extract_meta(
                pid, osp.join(seg_dir, seg), meta_csv)
            meta_dict.update({"path": {'img_path': osp.join(img_dir, img),
                                       "seg_path": osp.join(seg_dir, seg)}})
            self.update_info(pid, meta_dict)
        pass
