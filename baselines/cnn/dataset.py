# stanford CT dataset, moved from VAE

""" CT Dataset for NSCLC """
import sys
from collections import Counter, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae")

from datasets.patch.patch_stanfordradiogenomics import \
    StanfordRadiogenomicsPatchAugDataset
from datasets.utils import sitk2tensor

# implement image + label dataset


class StanfordLabelDataset(StanfordRadiogenomicsPatchAugDataset):

    def __init__(self, label_instance, split="all", transform=sitk2tensor, *args, **kwargs):
        if isinstance(split, str):
            super().__init__(split=split, transform=transform, *args, **kwargs)
        elif isinstance(split, (Sequence, np.ndarray)):
            super().__init__(transform=transform, *args, **kwargs)
            self.reset_split(split)
        self.label = label_instance
        self.values = [t[1] for t in self.label.regroup_tuples]
        pass

    def reset_split(self, split):
        assert isinstance(split, (Sequence, np.ndarray)), f"split must be a array or sequence but got {type(split)}"
        patches_list = np.array(sorted(self._get_img_files()))
        patient_list = self._get_patient_list(patches_list)
        patients = np.array(sorted(patient_list))[split]
        self.patches = self._get_patch_names(patients, patches_list)
        print(f"redo split: {len(split)=}, {len(self.patches)=}")
        pass

    def __getitem__(self, idx):
        img, name = super().__getitem__(idx)
        label = self.label.match_label(name)
        label = F.one_hot(torch.tensor(self.values.index(label)),
                          num_classes=len(self.values)).to(torch.float)
        return img, label


def generate_split(label_instance,
                   file_names,
                   seed=9001, fold=5):
    # match labels
    y_raw = label_instance.match_labels(file_names)
    Y = np.array(y_raw)
    # deal with NAs
    idx = np.arange(len(Y))
    na_idx = np.where(Y == "NA")[0]
    Y = np.delete(Y, na_idx, axis=0)
    idx = np.delete(idx, na_idx, axis=0)
    # stratified split
    skf = StratifiedKFold(n_splits=fold,
                          shuffle=True,
                          random_state=seed)

    # prepare the results
    for train_idx, test_idx in skf.split(idx, Y):
        name_train, name_test = np.array(
            file_names)[train_idx], np.array(file_names)[test_idx]
        # with index counting the NAs
        idx_train, idx_test = idx[train_idx], idx[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        print("train", Counter(y_train), "test", Counter(y_test))
        # yield the index
        yield (idx_train, idx_test), (name_train, name_test)
    pass
