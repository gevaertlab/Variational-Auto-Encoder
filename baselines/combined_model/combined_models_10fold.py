import os
import sys
import numpy as np
import pandas as pd
import os.path as osp
import json

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def main():
    fold = 10
    # load the pyra features
    feature_df = pd.read_csv(
        "/labs/gevaertlab/users/yyhhli/code/vae/baselines/pyradiomics/pyradiomics_features_default.csv")
    save_root = "/labs/gevaertlab/users/yyhhli/code/vae/baselines/combined_model/results/"
    numeric_cols = feature_df.select_dtypes(include=["int", "float"]).columns
    Xdf = feature_df[numeric_cols]
    X = Xdf.values

    # get file names
    # image_paths = feature_df["Image"].values
    # file_names = [osp.basename(path).split("_")[0] for path in image_paths]

    from torch.utils.data.dataloader import DataLoader
    from evaluations.export import Exporter
    from datasets.utils import sitk2tensor
    from datasets import PATCH_DATASETS
    from applications.cv_models import cv_predict_eval_with_model
    from datasets.label.label_stanfordradiogenomics import (
        LabelStfAJCC, LabelStfEGFRMutation, LabelStfHisGrade, LabelStfKRASMutation,
        LabelStfNStage, LabelStfReGroup, LabelStfRGLymphInvasion,
        LabelStfRGPleuralInvasion, LabelStfTStage)
    
    stfrg_patch = PATCH_DATASETS["StanfordRadiogenomicsPatchAugDataset"](root_dir=None,
                                                                         transform=sitk2tensor,
                                                                         split='all')
    stfrg_dl = DataLoader(dataset=stfrg_patch,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False,
                          num_workers=4,
                          pin_memory=True)
    label_list = [
        LabelStfTStage, LabelStfNStage, LabelStfAJCC, LabelStfHisGrade,
                  LabelStfRGPleuralInvasion, LabelStfEGFRMutation, LabelStfKRASMutation,
                  LabelStfRGLymphInvasion]
    label_instance_list = [l() for l in label_list]
    label_dict = {l.name: l for l in label_instance_list}
    task_names = list(label_dict.keys())
    tasks = ["StfTStage", "StfNStage", "StfAJCC", "StfHisGrade", "StfRGPleuralInvasion",
             "StfEGFRMutation", "StfKRASMutation", "StfRGLymphInvasion"]
    exporter = Exporter(base_model_name="VAE3D",
                        log_name="VAE3D32AUG",
                        version=70,
                        task_names=tasks,
                        dataloaders={"stf_all": stfrg_dl})
    embeddings, data_names = exporter.get_embeddings()
    # concat the features
    x_concat = np.concatenate([np.array(embeddings['stf_all']), X], axis=1)
    print(x_concat.shape)
    for i, task in enumerate(tasks):
        labels = exporter.get_labels(label_name=task,
                                     label_kwds={},
                                     data_names=data_names)
        not_na = (np.array(labels['stf_all']) != "NA")
        results = cv_predict_eval_with_model(task_type="classification",
                                             X=np.concatenate([np.array(embeddings['stf_all']), X], axis=1,)[
                                                 not_na],
                                             Y=np.array(labels['stf_all'])[
                                                 not_na],
                                             fold=fold,
                                             model_name="xgboost",
                                             tune_hparams=True,
                                             verbose=True,
                                             seed=9001)
        # postprocessing results
        organized_results = [r[0] for r in results]
        organized_results = {k: [dic[k] for dic in organized_results]
                             for k in organized_results[0].keys()}
        # save results as json
        with open(osp.join(save_root, f"{task_names[i]}_results_fold{fold}.json"), "w") as f:
            json.dump(organized_results, f)
    pass

if __name__ == "__main__":
    main()
    pass