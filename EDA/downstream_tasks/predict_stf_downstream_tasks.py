import argparse
import os
import sys
sys.path.insert(1, os.getcwd())

if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    from applications.application import Application
    from datasets.utils import sitk2tensor
    from datasets import PATCH_DATASETS
    parser = argparse.ArgumentParser(
        description='Eval downstream tasks in STF dataset')
    parser.add_argument("--tasks", "-t", nargs="+", default=["StfAJCC", "StfHisGrade",
                                                             "StfNStage", "StfTStage", "StfLymphInvasion",
                                                             "StfEGFRMutation", "StfKRASMutation"])
    parser.add_argument("--versions", "-v", nargs="+",
                        default=[49, 51, 53, 60])
    parser.add_argument("--log_name", "-ln", default="VAE3D32AUG")
    args = parser.parse_args()

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

    # "StfVolume"
    for task in args.tasks:
        for version in args.versions:
            print(
                f"======= Predicting {task} with model version {version} =======")
            app = Application(log_name=args.log_name,
                              version=version,
                              task_name=task,
                              task_kwds={"task_type": "classification"},
                              base_model_name='VAE3D',
                              dataloaders={'train': stfrg_train_patch_dataloader,
                                           'val': stfrg_test_patch_dataloader})

            result_dict, pred_dict, pred_stats, hparam_dict = app.task_prediction(
                tune_hparams=True, models='all')
    pass
