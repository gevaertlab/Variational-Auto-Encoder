# defines a pytorch lightning module that handles the training and validation of the CNN model

import json
import os
import numpy as np
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import StanfordLabelDataset, generate_split
from model import CNNClassification

from applications.models import CLASSIFICATION_METRIC_DICT
from datasets.label.label_stanfordradiogenomics import (
    LabelStfAJCC, LabelStfEGFRMutation, LabelStfHisGrade, LabelStfKRASMutation,
    LabelStfNStage, LabelStfReGroup, LabelStfRGLymphInvasion,
    LabelStfRGPleuralInvasion, LabelStfTStage)


class CNNModule(pl.LightningModule):

    def __init__(self, model, params: dict):
        super(CNNModule, self).__init__()
        if isinstance(model, dict):
            model = CNNClassification(**model)
        self.model = model

        self.params = params

        # init dataset
        self.dataset = StanfordLabelDataset
        pass

    def train_dataloader(self):
        self.train_ds = self.dataset(label_instance=self.params["data_params"]["label_instance"],
                                     split=self.params["data_params"]["train_idx"])
        return DataLoader(self.train_ds, **self.params["dataloader_params"])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        pass

    def val_dataloader(self):
        self.val_ds = self.dataset(label_instance=self.params["data_params"]["label_instance"],
                                   split=self.params["data_params"]["val_idx"])
        return DataLoader(self.val_ds, **self.params["dataloader_params"])

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def loss(self, preds, labels):
        """ cross entropy loss of pytorch """
        return F.cross_entropy(preds, labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss(preds, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["lr"])
        return [optimizer]

def main(params):
    model = CNNModule(params["model"], params)
    trainer = pl.Trainer(**params["trainer_params"])
    trainer.fit(model)
    # evaluation of the model
    results = {}
    y_val, y_hat_val = np.array([]), np.array([])
    for x, y in model.val_dataloader():
        y_hat = model(x)
        # evaluations of the model
        # convert to numpy array
        y_hat = y_hat.detach().cpu().numpy()[: ,1] > 0.5
        y = y.detach().cpu().numpy()[: ,1]
        # concat y and y hat
        y_val = np.concatenate((y_val, y))
        y_hat_val = np.concatenate((y_hat_val, y_hat))


    # NOTE: currently only handles binary classification
    for metric_name, metric_func in CLASSIFICATION_METRIC_DICT.items():
        metric = metric_func(y_val, y_hat_val)
        print(f"{metric_name}: {metric}")
        results[metric_name] = metric
    return results

def get_params():
    """ set default params and also allow to override them with args """
    params = {
        "model": {
            "n_classes": 2,
        },
        "lr": 1e-3,
        "trainer_params": {
            "accelerator": "gpu",
            "gpus": 1,
            "max_epochs": 50, # 500
            "auto_select_gpus": True,
            "log_every_n_steps": 10, },
        "data_params": {},
        "dataloader_params": {
            "batch_size": 32,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 4, }
    }
    return params


if __name__ == "__main__":
    params = get_params()
    print(params)
    # get the label instance
    seg_file_names = os.listdir(
        "/labs/gevaertlab/data/lung cancer/StanfordRadiogenomics/segmentations_nrrd")
    label_classes = [ # LabelStfAJCC, LabelStfEGFRMutation, LabelStfHisGrade, LabelStfKRASMutation, LabelStfNStage,
                     # LabelStfRGLymphInvasion, LabelStfRGPleuralInvasion,
                     LabelStfTStage]
    for label_class in label_classes:
        label_instance = label_class()
        params["model"]["n_classes"] = set(label_instance.regroup_dict.values()).__len__()
        params["data_params"]["label_instance"] = label_instance
        params["data_params"]["dataset_name"] = label_instance.name
        results = {n: [] for n in CLASSIFICATION_METRIC_DICT.keys()}
        for (idx_train, idx_test), (name_train, name_test) in generate_split(label_instance, file_names=seg_file_names):
            params["data_params"]["train_idx"] = idx_train
            params["data_params"]["val_idx"] = idx_test
            params["data_params"]["train_name"] = name_train
            params["data_params"]["val_name"] = name_test
            result = main(params)
            for k, v in result.items():
                results[k].append(v)
        print("label: ", label_instance.name)
        print("results: ", results)
        # save results
        with open(f"/labs/gevaertlab/users/yyhhli/code/vae/baselines/cnn/results/{label_instance.name}.json", "w") as f:
            json.dump(results, f)

    pass
