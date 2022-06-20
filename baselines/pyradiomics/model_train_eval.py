

import os
import os.path as osp
import sys
import time
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae")


from applications.models import (CLASSIFICATION_METRIC_DICT,
                                 CLASSIFICATION_MODELS, NJOB_DICT, RANDOM_DICT)
from datasets.label.label_stanfordradiogenomics import (
    LabelStfAJCC, LabelStfEGFRMutation, LabelStfHisGrade, LabelStfKRASMutation,
    LabelStfNStage, LabelStfReGroup, LabelStfRGLymphInvasion,
    LabelStfRGPleuralInvasion, LabelStfTStage)

def predict_task(X, file_names, label, seed=9001, fold=5):
    # match labels
    y_raw = label.match_labels(file_names)
    # deal with NAs
    Y = np.array(y_raw)
    na_idx = np.where(Y == "NA")[0]
    # exclude them in both X and Y
    X = np.delete(X, na_idx, axis=0)
    Y = np.delete(Y, na_idx, axis=0)

    # stratified split
    skf = StratifiedKFold(n_splits=fold,
                          shuffle=True,
                          random_state=seed)
    # prepare the results
    results = []
    for train_idx, test_idx in skf.split(X, Y):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        print("train", Counter(y_train), "test", Counter(y_test))
        # predict and evaluate
        model = CLASSIFICATION_MODELS["xgboost"]["basemodel"]().fit(
            x_train, y_train)
        y_pred = model.predict(x_test)
        r = {}
        for k, f in CLASSIFICATION_METRIC_DICT.items():
            if k not in ["AUROC", "AUPRC"]:
                r[k] = f(y_test, y_pred)
        results.append(r)
    # convert result to dataframe
    result_df = pd.DataFrame(results)
    return result_df


def main(result_dir="./results/",):

    # list out all the STF labels imported
    label_list = [
        # LabelStfTStage, LabelStfNStage, LabelStfAJCC, LabelStfHisGrade,
                #   LabelStfRGPleuralInvasion, LabelStfEGFRMutation, LabelStfKRASMutation,
                  LabelStfRGLymphInvasion]
    label_instance_list = [l() for l in label_list]
    label_dict = {l.name: l for l in label_instance_list}
    # load the features
    feature_df = pd.read_csv(
        "/labs/gevaertlab/users/yyhhli/code/vae/baselines/pyradiomics/pyradiomics_features_default.csv")
    numeric_cols = feature_df.select_dtypes(include=["int", "float"]).columns
    Xdf = feature_df[numeric_cols]
    X = Xdf.values

    # get file names
    image_paths = feature_df["Image"].values
    file_names = [osp.basename(path).split("_")[0] for path in image_paths]
    for lname, label in label_dict.items():
        print(label.name)
        results = predict_task(X, file_names, label, seed=9001, fold=5)
        # print the results
        print(results)
        # save the results
        results.to_csv(osp.join(result_dir, f"pyradiomics_default_results_xgboost_{lname}.csv"), index=False)
    pass


if __name__ == "__main__":
    main()
