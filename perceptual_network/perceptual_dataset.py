from dataset import LIDCPatch32Dataset
import SimpleITK as sitk
import os

from tasks import TaskVolume

class LIDCPatch32VolumeDataset(LIDCPatch32Dataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
    