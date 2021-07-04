""" 
call preprocess module 
"""

from patch_extraction.patch_extract import PatchExtract
from patch_extraction.registration.lidc import LidcReg

if __name__ == "__main__":
    augmentation_params = [
        ([
            ('shift',
             {'range': (-5, 5)})
        ],
            5)
    ]
    lidc_register = LidcReg()
    dataset_params = lidc_register.load_json()
    save_dir = "/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-patch-32_aug"
    vis_dir = "/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-visualization-32_aug"
    patch_extract = PatchExtract(patch_size=(32, 32, 32),
                                 augmentation_params=augmentation_params)
    patch_extract.extract_ds(dataset_params,
                             save_dir,
                             multi=True)
    patch_extract.vis_ds(save_dir,
                         vis_dir=vis_dir)
