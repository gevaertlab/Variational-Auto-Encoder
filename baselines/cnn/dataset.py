# stanford CT dataset, moved from VAE

""" CT Dataset for NSCLC """
from collections import Counter
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae")

from datasets.patch.patch_stanfordradiogenomics import StanfordRadiogenomicsPatchAugDataset

# implement image + label dataset


class StanfordLabelDataset(StanfordRadiogenomicsPatchAugDataset):

    def __init__(self, label_instance, split=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label_instance
        if split is not None:
            self.patches = np.array(sorted(self._get_img_files()))[split]
            print(f"redo split: {len(split)=}")
        pass

    def __getitem__(self, idx):
        img, name = super().__getitem__(idx)
        label = self.label.match_label(name)
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
        name_train, name_test = np.array(file_names)[train_idx], np.array(file_names)[test_idx]
        idx_train, idx_test = idx[train_idx], idx[test_idx] # with index counting the NAs
        y_train, y_test = Y[train_idx], Y[test_idx]
        print("train", Counter(y_train), "test", Counter(y_test))
        # yield the index
        yield (idx_train, idx_test), (name_train, name_test)
    pass
