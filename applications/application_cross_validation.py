""" cross validation evaluation of downstream tasks """


from applications.application import Application
from typing import Dict
from applications.cv_models import cv_predict_task
import os.path as osp
from utils.save_dict import JsonDict


class ApplicationCV(Application):
    """Use embedding + labels

    Args:
        Application ([type]): [description]
    """

    def __init__(self,
                 log_name: str,
                 version: int,
                 task_name: str,
                 task_kwds: dict = ...,
                 base_model_name: str = 'VAE3D',
                 dataloaders: Dict = ...):
        super().__init__(log_name,
                         version,
                         task_name,
                         task_kwds,
                         base_model_name,
                         dataloaders)
        pass

    def _init_results(self):
        # results: json dicts
        result_log_file = osp.join(self.save_dir,
                                   '.'.join([self.task_name, 'json']))
        osp.join(self.APP_DIR, result_log_file)
        self.result_dict = JsonDict(save_path=osp.join(self.APP_DIR,
                                                       self.save_dir,
                                                       '.'.join([self.task_name,
                                                                 'cv_result_dict',
                                                                 'json'])))
        # also initializing hparam_dict, in case we don't want to tune hparams
        self.hparam_dict = JsonDict(save_path=osp.join(self.APP_DIR,
                                                       self.save_dir,
                                                       '.'.join([self.task_name,
                                                                 'best_hparams',
                                                                 'json'])))
        pass

    def task_prediction(self,
                        tune_hparams=True,
                        models='xgboost',
                        fold=5):
        border = "-----"
        self.logger.info(
            f"{border}CV prediction for task {self.task_name}{border}")

        self.load_hparam_dict()
        results = self.result_dict
        results = cv_predict_task(task=self.task,
                                  X=self.embeddings, Y=self.labels,
                                  models=models,
                                  results=results,
                                  tune_hparams=tune_hparams,
                                  hparam_dict=self.hparam_dict,
                                  fold=fold,)

        return results

    def save_results(self, verbose=True):
        """ save predicted results """
        assert self.result_dict.keys(), "result_dict doesn't exist"
        self.result_dict.save()
        if verbose:
            self.logger.info(f"Saved results to {self.result_dict.save_path}")
        pass
