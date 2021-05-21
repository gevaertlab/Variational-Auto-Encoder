from dataset import LIDCPatch32Dataset
import SimpleITK as sitk
import os

from tasks import TaskVolume


# TODO implement other tasks and optimize data matching
class LIDCPatch32VolumeDataset(LIDCPatch32Dataset):

    LOG_DIR = f"{os.getcwd()}/logs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = "Y_matched_malignancy.npy"

    def _init_labels(self):
        """ loading labels """
        # either load labels or match labels and save
        if os.path.exists(os.path.join(self.LOG_DIR), self.file_name):
            
        pass

    def load_labels(self):
        """ match labels """
        
        pass

    def __getitem__(self, idx: int):
        img_name = self.img_lst[idx]
        img = sitk.ReadImage(os.path.join(self.root_dir, img_name))
        sample = {'image': img, 'image_name': img_name}
        # apply transformation
        if self.transform is not None:
            sample = self.transform(sample)
        tv = TaskVolume()
        volume = tv.getLabel(sample['image_name'].replace('.nrrd', ''))
        return sample['image'], volume
