'''
extract patches:
    preprocessing
    augmentation
    crop
    save
'''

import os
import os.path as osp
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from tqdm import tqdm
from utils.funcs import mkdir_safe
from utils.io import load_dcm, load_nrrd, save_as_nrrd
from utils.visualization import vis_sitk

from .augmentation import Augmentation, calc_crop_size
from .crop import crop_patch
from .preprocess_funcs import preprocess
from datasets.ct import CTDataset


class PatchExtract:

    def __init__(self,
                 patch_size: Tuple,
                 dataset: CTDataset,
                 augmentation_params=None,
                 debug=False):
        self.patch_size = patch_size
        self.dataset = dataset
        # self.ds_params = self.dataset.generate_ds_params()  # deprecated
        self.debug = debug  # debug flag, whether to output example images
        self.augmentation_params = augmentation_params
        if augmentation_params:
            self.augmentation = Augmentation(augmentation_params)
        else:
            self.augmentation = None
        pass

    def extract_img(self,
                    img: np.ndarray,
                    center_point: Union[list, Tuple],
                    save_path: str):
        """extract patchES from a single image
        1. preprocess
        2. augment
        3. crop
        4. save
        Args:
            img ([np.array]): [3D]
            center_point ([tuple]): [coord of the nodule]
            save_path ([str]): [path to save this patch]
        """
        spacing, transpose_axis = self.dataset.get_info('spacing'), \
            self.dataset.get_info('transpose_axis')
        # preprocesing: np_image, new center point
        img, center_point = preprocess(img=img,
                                       center_point=center_point,
                                       spacing=spacing,
                                       transpose_axis=transpose_axis)
        # augment
        if self.augmentation:
            # pre-crop
            new_size = calc_crop_size(self.patch_size,
                                      self.augmentation_params)
            img = crop_patch(img, center_point, size=new_size)
            center_point = tuple(int(p/2) for p in new_size)

            # generate aug
            aug_gen = self.augmentation.augment_generator(img, center_point)
            for i, (aug_img, aug_cp) in enumerate(aug_gen):
                # crop
                patch = crop_patch(aug_img, aug_cp, size=self.patch_size)
                # save
                save_as_nrrd(patch,
                             save_path.replace(
                                 '.nrrd', f'.Aug{str(i).zfill(2)}.nrrd'),
                             verbose=1)
        else:
            patch = crop_patch(img, center_point, size=self.patch_size)
            save_as_nrrd(patch,
                         save_path,
                         verbose=1)
        pass

    def load_extract(self,
                     item,
                     save_dir: str,
                     overwrite=False,):
        meta = self.dataset.get_info(item)[1]
        img_path = meta['path']
        centroid_dict = meta['centroid_dict']
        for k, v in centroid_dict.items():
            file_name = f"{meta['pid']}.{str(k)}.nrrd"
            img = self.dataset.load_funcs['ct'](img_path)
            save_path = osp.join(save_dir, file_name)
            if not overwrite and os.path.exists(save_path):
                self.logger.info("{save_path} already exists")
                continue
            self.extract_img(img=img,
                             center_point=v,
                             save_path=save_path)
        pass

    # def load_extract(self,
    #                  img_path,
    #                  load_func,
    #                  center_point,
    #                  file_name,
    #                  save_dir):
    #     # calls load image and extract image
    #     img = self.dataset[img_path]
    #     # img = self.load_img(img_path=img_path,
    #     #                     load_func=load_func)
    #     self.extract_img(img,
    #                      center_point,
    #                      osp.join(save_dir, file_name))
    #     pass

    # def load_img(self, img_path=None, load_func=None):
    #     if not load_func:
    #         return load_dcm(img_path)
    #     else:
    #         return load_func(img_path)

    def load_extract_ds(self,
                        save_dir,
                        overwrite=False,
                        multi=False):
        # if not multi: # extract one by one
        #     dataloader = DataLoader()

        # TODO: re-implement the whole class to use the dataset -> dataloader
        # transform format to load extract

        # called load_extract for the whole dataset
        load_func = self.dataset.load_funcs['ct']
        if not multi:
            
            # for file_name, (img_path, center_point) in tqdm(self.dataset.get_info('data_dict').items()):
            #     self.load_extract(img_path=img_path,
            #                       load_func=load_func,
            #                       center_point=tuple(center_point),
            #                       file_name=file_name+'.nrrd',  # add ext
            #                       save_dir=save_dir)
            
            for i in tqdm(range(len(self.dataset))):
                self.load_extract(item=i, save_dir=save_dir, overwrite=overwrite)
        else:
            data_tuple = [
                (img_path,
                 load_func,
                 tuple(center_point),
                 file_name+'.nrrd',
                 save_dir) for
                file_name, (img_path, center_point)  # have to redo this
                in self.ds_params['data_dict'].items()
            ]
            with Pool(cpu_count()) as p:
                p.starmap(self.load_extract,
                          tqdm(data_tuple,
                               total=len(data_tuple)))

        pass

    def vis_debug(self):
        # when called, will run some debug visualization functions.
        raise NotImplementedError

    def vis_ds(self, dataset_dir, vis_dir=None, rd=20):
        # called when debug flag present
        if vis_dir is None:  # default path
            vis_dir = str(Path(dataset_dir).parent.absolute())
        mkdir_safe(vis_dir)
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.nrrd')]
        if rd:  # random sampling
            files = random.sample(files, rd)
        for file in files:
            self.vis_img(img_path=os.path.join(dataset_dir, file),
                         vis_path=os.path.join(vis_dir,
                                               file.replace('.nrrd', '.jpeg')))
        pass

    def vis_img(self, img_path, vis_path):
        # called when debug flag present.
        img = load_nrrd(img_path)
        vis_sitk(img, vis_path=vis_path)
        pass
