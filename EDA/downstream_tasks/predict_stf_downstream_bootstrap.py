import sys
import argparse

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def main():
    from torch.utils.data.dataloader import DataLoader
    from applications.application import Application
    from datasets.utils import sitk2tensor
    from datasets import PATCH_DATASETS

    parser = argparse.ArgumentParser(
        description='Eval downstream tasks in STF dataset using bootstrap')
    parser.add_argument("--tasks", "-t", nargs="+", default=["StfAJCC", "StfHisGrade",
                                                             "StfNStage", "StfTStage", "StfLymphInvasion",
                                                             "StfEGFRMutation", "StfKRASMutation"])
    parser.add_argument("--versions", "-v", nargs="+",
                        default=[1, 2, 3])
    parser.add_argument("--log_name", "-ln", default="PRETRAINED_VAE")
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

    # "StfAJCC", "StfHisGrade", "StfNStage", "StfTStage",
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
            if version == 60: # NOTE: only tune parameters for the first version
                hp = True
            else:
                hp = False
            result_dict, hparam_dict = app.task_prediction(
                tune_hparams=hp, models='random_forest', bootstrapping=True)

    # for task in ["StfHisGrade", "StfNStage", "StfTStage", "StfLymphInvasion", "StfEGFRMutation", "StfKRASMutation"]:  # "StfAJCC",
    #     for version in [1, 2, 3]:
    #         print(
    #             f"======= Predicting {task} with model version {version} =======")
    #         app = Application(log_name='PRETRAINED_VAE',
    #                           version=version,
    #                           task_name=task,
    #                           task_kwds={"task_type": "classification"},
    #                           base_model_name='VAE3D',
    #                           dataloaders={'train': stfrg_train_patch_dataloader,
    #                                        'val': stfrg_test_patch_dataloader})
    #         if version == 60:
    #             hp = True
    #         else:
    #             hp = False
    #         result_dict, hparam_dict = app.task_prediction(
    #             tune_hparams=hp, models='random_forest', bootstrapping=True)
    pass


if __name__ == "__main__":
    main()
