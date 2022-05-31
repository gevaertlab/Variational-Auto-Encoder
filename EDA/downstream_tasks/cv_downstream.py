import os
import sys

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def main():
    from torch.utils.data.dataloader import DataLoader
    from applications.application_cross_validation import ApplicationCV
    from datasets.utils import sitk2tensor
    from datasets import PATCH_DATASETS

    stfrg_train_patch = PATCH_DATASETS["StanfordRadiogenomicsPatchAugDataset"](root_dir=None,
                                                                               transform=sitk2tensor,
                                                                               split='train')
    stfrg_train_patch_dataloader = DataLoader(dataset=stfrg_train_patch,
                                              batch_size=1,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=4,
                                              pin_memory=True)
    stfrg_test_patch = PATCH_DATASETS["StanfordRadiogenomicsPatchDataset"](root_dir=None,
                                                                           transform=sitk2tensor,
                                                                           split='test')
    stfrg_test_patch_dataloader = DataLoader(dataset=stfrg_test_patch,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=4,
                                             pin_memory=True)

    app = ApplicationCV(log_name="VAE3D32AUG",
                        version=70,
                        task_name="StfAJCC",
                        task_kwds={"task_type": "classification"},
                        base_model_name='VAE3D',
                        dataloaders={'train': stfrg_train_patch_dataloader,
                                     'val': stfrg_test_patch_dataloader})

    results = app.task_prediction(tune_hparams=True,models="xgboost")
    print(results)
    app.save_results(verbose=True)
    pass

if __name__ == "__main__":
    main()
    pass