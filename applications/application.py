''' This file utilize the trained encoder to do downstream tasks '''


from datasets.embedding import EmbeddingPredictor, Embedding
from evaluations.export import Exporter
from utils.python_logger import get_logger
from utils.save_dict import NpyDict
from utils.save_dict import JsonDict
from configs.config_vars import BASE_DIR
from .__init__ import TASK_DICT
from datasets.label.label_dict import LABEL_DICT
import os
import os.path as osp
from .models import predictTask
from utils.visualization import vis_clustermap, vis_heatmap, ytrue_ypred_scatter, confusion_matrix_models, vis_pca, vis_tsne
from utils.timer import Timer


class Application:
    """
    Use embedding + label to do predictions
    input: param to get embedding; name of label task
    do: get embedding; use embedding to get labels; predict and save results
    output: results
    """

    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    APP_DIR = os.path.join(BASE_DIR, 'applications')

    def __init__(self,
                 log_name: str,
                 version: int,
                 task_name: str,
                 base_model_name: str = 'VAE3D'):
        self.timer = Timer(
            name=(osp.basename(__file__), self.__class__.__name__))
        self.timer()
        # logger
        self.logger = get_logger(self.__class__.__name__)
        # to get embedding
        self.base_model_name = base_model_name
        self.log_name = log_name
        self.version = version

        # get load and save dir
        self.load_dir = os.path.join(self.LOG_DIR,
                                     log_name,
                                     f'version_{version}')  # NOTE
        self.save_dir = os.path.join(self.APP_DIR,
                                     "results",
                                     f"{self.log_name}_{self.version}")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # to get label and task
        self.task_name = task_name

        # calculations:
        self.exporter = Exporter(base_model_name=base_model_name,
                                 log_name=log_name,
                                 version=version)
        self.embeddings, self.data_names = self.exporter.get_embeddings()
        self.labels = self.exporter.get_labels(
            task_name, data_names=self.data_names)
        self.task = TASK_DICT[task_name]()

        # init: results
        self._init_results()
        self.timer("initializing")
        pass

    def _init_results(self):
        # results: json dicts
        result_log_file = osp.join(self.save_dir,
                                   '.'.join([self.task_name, 'json']))
        osp.join(self.APP_DIR, result_log_file)
        self.result_dict = JsonDict(save_path=osp.join(self.APP_DIR,
                                                       self.save_dir,
                                                       '.'.join([self.task_name,
                                                                 'result_dict',
                                                                 'json'])))
        self.pred_dict = NpyDict(save_path=osp.join(self.APP_DIR,
                                                    self.save_dir,
                                                    '.'.join([self.task_name,
                                                              'preds',
                                                              'npy'])))
        self.hparam_dict = JsonDict(save_path=osp.join(self.APP_DIR,
                                                       self.save_dir,
                                                       '.'.join([self.task_name,
                                                                 'best_hparams',
                                                                 'json'])))
        self.pred_stats = NpyDict(save_path=osp.join(self.APP_DIR,
                                                     self.save_dir,
                                                     '.'.join([self.task_name,
                                                               'pred_stats',
                                                               'npy'])))
        pass

    # def get_labels(self, label_name):
    #     """
    #     get labels assuming embedding is get,
    #     return dict of labels not label instance
    #     """
    #     label_instance = LABEL_DICT[label_name]()
    #     label = {}

    #     # get labels
    #     self.timer()
    #     label['train'] = label_instance.get_labels(self.data_names['train'])
    #     label['val'] = label_instance.get_labels(self.data_names['val'])
    #     self.timer('match labels')
    #     return label

    # def get_embeddings(self, augment=False):
    #     self.logger.info("initializing embeddings")
    #     embeddings = {'train': Embedding(self.log_name,
    #                                      self.version,
    #                                      split='train'),
    #                   'val': Embedding(self.log_name,
    #                                    self.version,
    #                                    split='val')}

    #     predictor = EmbeddingPredictor(self.base_model_name,
    #                                    self.log_name,
    #                                    self.version)

    #     if not embeddings['train'].saved:
    #         embeddings['train'] = predictor.predict_embedding(dataloader='train_dataloader',
    #                                                           split='train')
    #     else:
    #         _ = embeddings['train'].load()
    #     if not embeddings['val'].saved:
    #         embeddings['val'] = predictor.predict_embedding(dataloader='val_dataloader',
    #                                                         split='val')
    #     else:
    #         _ = embeddings['val'].load()

    #     self.embeddings['train'], self.data_names['train'] = \
    #         embeddings['train'].get_embedding(augment=augment)
    #     self.embeddings['val'], self.data_names['val'] = \
    #         embeddings['val'].get_embedding(augment=augment)

    #     return self.embeddings, self.data_names

    def preprocess_data(self):
        # format the X and Y so that they can be taken by sklearn models
        X, Y = self.task.transform(self.embeddings, self.labels)
        return X, Y

    def task_prediction(self, tune_hparams=True, models='all'):
        """
        Predict a task
        Args:
            models (str or list of str, optional): models to use. Defaults to 'all'.
        Returns:
            dict: results (metrics) of predictions
        """
        border = "-----"
        self.logger.info(
            f"{border}Prediction for task {self.task_name}{border}")

        # TRAIN + PREDICT
        self.load_hparam_dict()
        results = (self.result_dict,
                   self.pred_dict,
                   self.pred_stats,
                   self.hparam_dict)
        results = predictTask(task=self.task,
                              X=self.embeddings, Y=self.labels,
                              models=models,
                              results=results,
                              tune_hparams=tune_hparams,
                              hparam_dict=self.hparam_dict)
        result_dict, pred_dict, pred_stats, hparam_dict = results
        return result_dict, pred_dict, pred_stats, hparam_dict

    def association_analysis(self):
        """
        perform association analysis between embeddings and labels
        using non-auged images in training set
        """
        # TODO: implement this
        pass

    def load_hparam_dict(self):
        """ load hparam dictionary which should be the same place as save"""

        if os.path.exists(self.hparam_dict.save_path):
            self.logger.info("Loading best hparams ...")
            self.hparam_dict.load()
        else:
            self.logger.info("New task, no hparams file loaded")
        pass

    def save_results(self, verbose=True):
        """
        save the results after completing a task prediction job
        saving:
        1. metrics for best models
        2. hyperparameters for best models
        3. predictions for best models
        """

        # 1. saving metrics for best models
        assert self.result_dict.keys(), "result_dict doesn't exist"
        # save result_dict file
        self.result_dict.save()
        if verbose:
            self.logger.info(f"Saved results to {self.result_dict.save_path}")

        # 2. saveing hyperparameters for best models
        assert self.hparam_dict.keys(), "hparam_dict doesn't exist"
        # save hparam_dict file
        self.hparam_dict.save()
        if verbose:
            self.logger.info(f"Saved results to {self.hparam_dict.save_path}")

        # 3. saving predictions for best models: use NPY files
        assert self.pred_dict.keys(), "pred_dict doesn't exist"
        # save pred_dict file
        self.pred_dict.save()
        if verbose:
            self.logger.info(f"Saved results to {self.pred_dict.save_path}")

        # 4. saving pred_stats for best models: use NPY files
        # NOTE: is ok for the pred_stats to be not exist
        if self.pred_stats.keys():
            # save self.pred_stats file
            self.pred_stats.save()
            if verbose:
                self.logger.info(
                    f"Saved results to {self.pred_stats.save_path}")
            pass

    def load_results(self, verbose=True):
        """
        load the results after completing a task prediction job
        loading:
        1. metrics for best models
        2. hyperparameters for best models
        3. predictions for best models
        """
        self.logger.info("Loading results")
        # 1. saving metrics for best models
        assert osp.exists(
            self.result_dict.save_path), "result_dict doesn't exist"
        # save result_dict file
        self.result_dict.load()
        if verbose:
            self.logger.info(f"Load results from {self.result_dict.save_path}")

        # 2. saveing hyperparameters for best models
        assert osp.exists(
            self.hparam_dict.save_path), "hparam_dict doesn't exist"
        # save hparam_dict file
        self.hparam_dict.load()
        if verbose:
            self.logger.info(f"Load results from {self.hparam_dict.save_path}")

        # 3. saving predictions for best models: use NPY files
        assert osp.exists(self.pred_dict.save_path), "pred_dict doesn't exist"
        # save pred_dict file
        self.pred_dict.load()
        if verbose:
            self.logger.info(f"Load results from {self.pred_dict.save_path}")

        # 4. saving pred_stats for best models: use NPY files
        # NOTE: is ok for the pred_stats to be not exist
        if osp.exists(self.pred_stats.save_path):
            # save self.pred_stats file
            self.pred_stats.load()
            if verbose:
                self.logger.info(
                    f"Load results from {self.pred_stats.save_path}")
            pass

    def draw_dignosis_figure(self, verbose=True):
        assert self.pred_dict.keys(), "pred_dict not exists"
        # based on the task type, draw different things
        diagnosis_figure_file = os.path.join(
            self.save_dir, '.'.join([self.task_name, 'figure.jpeg']))
        if self.task.task_type == 'regression':
            # draw scatter
            try:
                ytrue_ypred_scatter(self.pred_dict, os.path.join(
                    self.APP_DIR, diagnosis_figure_file))
            except Exception as e:
                self.logger.info(e)
                return
        elif self.task.task_type == 'classification':
            try:
                confusion_matrix_models(self.pred_dict,
                                        os.path.join(self.APP_DIR,
                                                     diagnosis_figure_file),
                                        classes=list(range(1, 6)))
            except Exception as e:
                self.logger.info(e)
                return
        if verbose:
            self.logger.info(f"Saved figure to {diagnosis_figure_file}")
        pass

    def visualize(self):
        """ visualizing with PCA and t-SNE and heatmap for embeddings with label """
        X, Y = self.preprocess_data()
        save_dir = os.path.join(self.APP_DIR,
                                "visualizations")
        if self.task.task_type == 'regression':
            kwarg = {'label_numeric': True}
        else:
            kwarg = {}

        # train
        vis_clustermap({'features': X['train'], 'nodule': Y['train']},
                       xlabel='features', ylabel='nodule', task_name=self.task_name,
                       save_path=osp.join(save_dir,
                                          f"{self.version}_{self.task_name}_clustermap_train.jpeg"))
        # vis_heatmap(X['train'], save_path=os.path.join(
        #             save_dir, f"{self.version}_heatmap_train.jpeg"),
        #             xlabel='features', ylabel='nodule')
        # vis_pca(data=X['train'], label=Y['train'],
        #         save_path=os.path.join(
        #             save_dir, f"{self.version}_{self.task_name}_pca_train.jpeg"),
        #         label_name=self.task_name, **kwarg)
        # vis_tsne(data=X['train'], label=Y['train'],
        #          save_path=os.path.join(
        #              save_dir, f"{self.version}_{self.task_name}_pca_train.jpeg"),
        #          label_name=self.task_name, **kwarg)

        # val
        vis_clustermap({'features': X['train'], 'nodule': Y['train']},
                       xlabel='features', ylabel='nodule', task_name=self.task_name,
                       save_path=osp.join(save_dir,
                                          f"{self.version}_{self.task_name}_clustermap_test.jpeg"))
        # vis_heatmap(X['val'], save_path=os.path.join(
        #             save_dir, f"{self.version}_heatmap_val.jpeg"))
        # vis_pca(data=X['val'], label=Y['val'],
        #         save_path=os.path.join(
        #             save_dir, f"{self.version}_{self.task_name}_pca_val.jpeg"),
        #         label_name=self.task_name, **kwarg)
        # vis_tsne(data=X['val'], label=Y['val'],
        #          save_path=os.path.join(
        #              save_dir, f"{self.version}_{self.task_name}_pca_val.jpeg"),
        #          label_name=self.task_name, **kwarg)
        pass
