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

from tqdm import tqdm
from utils.funcs import mkdir_safe
from utils.io import load_dcm, load_nrrd, save_as_nrrd
from utils.visualization import vis_sitk

from .augmentation import Augmentation
from .crop import crop_patch
from .preprocess_funcs import preprocess


class PatchExtract:

    def __init__(self,
                 patch_size,
                 ds_params=None,
                 augmentation_params=None,
                 debug=False):
        self.patch_size = patch_size
        self.ds_params = ds_params
        self.debug = debug  # debug flag, whether to output example images
        if augmentation_params:
            self.augmentation = Augmentation(augmentation_params)
        else:
            self.augmentation = None
        pass

    def extract_img(self,
                    img,
                    center_point,
                    save_path):
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
        # preprocesing
        img, center_point = preprocess(img=img,
                                       center_point=center_point,
                                       spacing=self.ds_params['spacing'],
                                       transpose_axis=self.ds_params['transpose_axis'])
        # augment
        if self.augmentation:
            aug_gen = self.augmentation.augment_generator(img, center_point)
        for i, (aug_img, aug_cp) in enumerate(aug_gen):
            # crop
            patch = crop_patch(aug_img, aug_cp, size=self.patch_size)
            # save
            save_as_nrrd(patch,
                         save_path.replace(
                             '.nrrd', f'.Aug{str(i).zfill(2)}.nrrd'),
                         verbose=1)
        pass

    def load_extract(self,
                     img_path,
                     load_func,
                     center_point,
                     file_name,
                     save_dir):
        # calls load image and extract image
        img = self.load_img(img_path=img_path,
                            load_func=load_func)
        self.extract_img(img,
                         center_point,
                         osp.join(save_dir, file_name))
        pass

    def load_img(self, img_path=None, load_func=None):
        if not load_func:
            return load_dcm(img_path)
        else:
            return load_func(img_path)

    def load_extract_ds(self,
                        save_dir,
                        multi=False):
        # called load_extract for the whole dataset
        load_func = self.ds_params['load_func']
        if not multi:
            for file_name, (img_path, center_point) in tqdm(self.ds_params['data_dict'].items()):
                self.load_extract(img_path=img_path,
                                  load_func=load_func,
                                  center_point=tuple(center_point),
                                  file_name=file_name+'.nrrd',  # add ext
                                  save_dir=save_dir)
        else:
            data_tuple = [
                (img_path,
                 load_func,
                 tuple(center_point),
                 file_name+'.nrrd',
                 save_dir) for
                file_name, (img_path, center_point)
                in self.ds_params['data_dict'].items()
            ]
            with Pool(cpu_count()) as p:
                p.starmap(self.load_extract,
                          tqdm(data_tuple,
                               total=len(data_tuple)))
        # visualization

        pass

    def vis_debug(self):
        # when called, will run some debug visualization functions.
        raise NotImplementedError

    def vis_ds(self, dataset_dir, vis_dir=None, rd=20):
        # called when debug flag present
        if vis_dir is None:  # default path
            vis_dir = Path(dataset_dir).parent.absolute()

        mkdir_safe(vis_dir)
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.nrrd')]
        if rd:  # random sapmling
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
