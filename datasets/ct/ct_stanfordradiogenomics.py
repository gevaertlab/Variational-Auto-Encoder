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
                     "save_path": "/labs/gevaertlab/data/lungcancer/StanfordRadiogenomics/stf_info_dict.json"},
                 reset_info=False):

        if root_dir is None:
            root_dir = "/labs/gevaertlab/data/NSCLC_Radiogenomics/"  # HACK: hard code this
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
        folders = os.listdir(self.root_dir)
        filelist = []
        for folder in folders:
            files = os.listdir(osp.join(self.root_dir, folder))

            img = [f for f in files if f.endswith("_img.nii.gz")][0]
            seg = [f for f in files if f.endswith("_msk.nii.gz")][0]
            filelist.append((osp.join(folder, img), osp.join(folder, seg)))

        # img_dir = osp.join(self.root_dir, "Images")  # HACK: hard code
        # seg_dir = osp.join(self.root_dir, "Segmentations")
        # img_files = os.listdir(img_dir)
        # seg_dirs = os.listdir(seg_dir)

        # # NOTE: segmentation files changed

        # filelist = []
        # for imgf in img_files:
        #     # HACK: try to get the segmentation subdir
        #     seg_subdir = imgf.split("_")[0]
        #     if seg_subdir in seg_dirs:
        #         segfname = os.listdir(osp.join(seg_dir, seg_subdir))[0]
        #         segf = osp.join(seg_subdir, segfname)
        #         filelist.append((imgf, segf))

        return filelist

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

    @staticmethod
    def _get_volume_from_seg(seg_np):
        seg_binary = (seg_np != 0).astype("uint8")
        # NOTE: not numpy stuff, or you can save into json
        return int(np.sum(seg_binary))

    @staticmethod
    def _get_centroid_from_seg(seg_np):
        # 3D
        # has to be int (not numpy stuff) or cannot saved as json
        seg_bool = (seg_np > 0).astype('int')
        y = int(np.median(np.nonzero(
            (np.sum(seg_bool, axis=(1, 2)) > 0).astype('int'))))
        x = int(np.median(np.nonzero(
            (np.sum(seg_bool, axis=(0, 2)) > 0).astype('int'))))
        z = int(np.median(np.nonzero(
            (np.sum(seg_bool, axis=(0, 1)) > 0).astype('int'))))

        return {0: (x, y, z)}

    # def load_seg_np(self, seg_path):
    #     seg_itk = self.load_seg_itk(str(seg_path))
    #     return sitk.GetArrayFromImage(seg_itk).transpose((2, 1, 0))

    # def load_seg_itk(self, seg_path):
    #     return self.load_funcs['seg'](seg_path)

    def _set_ds_info(self):
        filelist = self._get_files()
        meta_csv = self._get_meta_csv()
        for img, seg in tqdm(filelist):
            pid = img.replace("_img.nii.gz", "")
            meta_dict = self._extract_meta(
                pid, osp.join(self.root_dir, seg), meta_csv)
            meta_dict.update({"path": {'img_path': osp.join(self.root_dir, img),
                                       "seg_path": osp.join(self.root_dir, seg)}})
            self.update_info(pid, meta_dict)
        pass
