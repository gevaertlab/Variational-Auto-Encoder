""" util funcs and class related to dataset modifications """

import SimpleITK as sitk
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import deque


def train_val_test_split(length: int,
                         ratio: float = 0.1,
                         random_state=9001):
    """
    Split dataset to train, val and test sets
    Args:
        length (int): length of the dataset to split (num_samples)
        ratio (float, optional): what's ratio of val and test set. Defaults to 0.1.
        random_state (int, optional): sklearn random_state. Defaults to 9001.

    Returns:
        array like: index for train, val and test set
    """
    indices = list(range(length))
    train_val_idx, test_idx = train_test_split(indices,
                                               test_size=ratio,
                                               random_state=random_state)
    train_idx, val_idx = train_test_split(train_val_idx,
                                          test_size=ratio/(1-ratio),
                                          random_state=random_state)
    return train_idx, val_idx, test_idx


class sitk2numpy(object):
    ''' Convert sitk image to numpy and add a channel as the first dimension (float numpy) '''

    def __init__(self):
        pass

    def __call__(self, sample):
        img, img_name = sample['image'], sample['image_name']
        npimg = sitk.GetArrayFromImage(img)
        npimg = np.expand_dims(npimg, axis=0).astype('float32')
        return {'image': npimg, 'image_name': img_name}


class np2tensor(object):
    ''' Convert numpy image to tensor (float tensor) '''

    def __init__(self):
        pass

    def __call__(self, sample):
        ''' 3D image, thus no need to swap axis '''
        img, img_name = sample['image'], sample['image_name']
        return {'image': torch.from_numpy(img), 'image_name': img_name}


sitk2tensor = transforms.Compose([sitk2numpy(), np2tensor()])


class LRUCache:
    def __init__(self, capacity: int):
        self.dict = {}
        self.queue = deque([])
        self.capacity = capacity

    def has(self, key):
        return (key in self.dict)

    def queue_update(self, key):
        # For every operation we want to update queue
        if key in self.queue:
            self.queue.remove(key)
        self.queue.append(key)

    def get(self, key: int):
        # TC is O(N) where N is no of instructions
        self.queue_update(key)
        return self.dict.get(key, -1)

    def put(self, key: int, value: int):
        # TC is same here O(N)
        # check if the key is not in dictionary already and if there are more than capacity items in dictionary
        if key not in self.dict.keys() and len(self.dict) >= self.capacity:
            while 1:  # keep doing this shit because the key might not exist in the dictionary
                try:
                    evacuate = self.queue.popleft()
                    del self.dict[evacuate]
                    break
                except Exception:
                    continue
        self.queue_update(key)
        self.dict[key] = value
        pass
