''' This file utilize the trained encoder to do downstream tasks '''


from datasets.embedding import EmbeddingPredictor, Embedding
from utils.save_dict import NpyDict
from utils.save_dict import JsonDict
from configs.config_vars import BASE_DIR
from .__init__ import TASK_DICT
from datasets.label.label_dict import LABEL_DICT
import os
import os.path as osp
from .models import predictTask
from utils.visualization import vis_heatmap, ytrue_ypred_scatter, confusion_matrix_models, vis_pca, vis_tsne
from utils.funcs import Timer


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
        self.timer = Timer()

        # calculations:
        embeddings, label_names = self.get_embeddings()
        self.labels = self.get_labels(task_name)
        self.task = TASK_DICT[task_name]()

        # init: results
        self._init_results()
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

    def get_labels(self, label_name):
        """ 
        get labels assuming embedding is get, 
        return dict of labels not label instance 
        """
        label_instance = LABEL_DICT[label_name]()
        label = {}

        # get labels
        self.timer()
        label['train'] = label_instance.get_labels(self.data_names['train'])
        label['val'] = label_instance.get_labels(self.data_names['val'])
        self.timer('match labels')
        return label

    def get_embeddings(self, split='train'):

        embeddings = {'train': Embedding(self.log_name,
                                         self.version,
                                         split='train'),
                      'val': Embedding(self.log_name,
                                       self.version,
                                       split='val')}

        predictor = EmbeddingPredictor(self.base_model_name,
                                       self.log_name,
                                       self.version)

        if not embeddings['train'].saved:
            embeddings['train'] = predictor.predict_embedding(dataloader='train_dataloader',
                                                              split='train')
        else:
            _ = embeddings['train'].load()
        if not embeddings['val'].saved:
            embeddings['val'] = predictor.predict_embedding(dataloader='val_dataloader',
                                                            split='val')
        else:
            _ = embeddings['val'].load()

        self.embeddings = {'train': embeddings['train']._data['embedding'],
                           'val': embeddings['val']._data['embedding']}
        self.data_names = {'train': embeddings['train']._data['index'],
                           'val': embeddings['val']._data['index']}
        return self.embeddings, self.data_names

    def preprocess_data(self):
        # format the X and Y so that they can be taken by sklearn models
        X, Y = self.task.transform(self.embeddings, self.labels)
        return X, Y

    def task_prediction(self, models='all'):
        """
        Predict a task
        Args:
            models (str or list of str, optional): models to use. Defaults to 'all'.
        Returns:
            dict: results (metrics) of predictions
        """
        border = "-----"
        print(f"{border}Prediction for task {self.task_name}{border}")
        X, Y = self.preprocess_data()

        # TRAIN + PREDICT
        self.load_hparam_dict()
        results = (self.result_dict,
                   self.pred_dict,
                   self.pred_stats,
                   self.hparam_dict)
        results = predictTask(X, Y,
                              self.task.task_type,
                              models=models,
                              results=results,
                              hparam_dict=self.hparam_dict)
        result_dict, pred_dict, pred_stats, hparam_dict = results
        return result_dict, pred_dict, pred_stats, hparam_dict

    def load_hparam_dict(self):
        """ load hparam dictionary which should be the same place as save"""

        if os.path.exists(self.hparam_dict.save_path):
            print("Loading best hparams ...")
            self.hparam_dict.load()
        else:
            print("New task, no hparams file loaded")
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
            print(f"Saved results to {self.result_dict.save_path}")

        # 2. saveing hyperparameters for best models
        assert self.hparam_dict.keys(), "hparam_dict doesn't exist"
        # save hparam_dict file
        self.hparam_dict.save()
        if verbose:
            print(f"Saved results to {self.hparam_dict.save_path}")

        # 3. saving predictions for best models: use NPY files
        assert self.pred_dict.keys(), "pred_dict doesn't exist"
        # save pred_dict file
        self.pred_dict.save()
        if verbose:
            print(f"Saved results to {self.pred_dict.save_path}")

        # 4. saving pred_stats for best models: use NPY files
        # NOTE: is ok for the pred_stats to be not exist
        if self.pred_stats.keys():
            # save self.pred_stats file
            self.pred_stats.save()
            if verbose:
                print(f"Saved results to {self.pred_stats.save_path}")
            pass

    def load_results(self, verbose=True):
        """
        load the results after completing a task prediction job
        loading:
        1. metrics for best models
        2. hyperparameters for best models
        3. predictions for best models
        """
        print("[Application] Loading results")
        # 1. saving metrics for best models
        assert osp.exists(
            self.result_dict.save_path), "result_dict doesn't exist"
        # save result_dict file
        self.result_dict.load()
        if verbose:
            print(f"Load results from {self.result_dict.save_path}")

        # 2. saveing hyperparameters for best models
        assert osp.exists(
            self.hparam_dict.save_path), "hparam_dict doesn't exist"
        # save hparam_dict file
        self.hparam_dict.load()
        if verbose:
            print(f"Load results from {self.hparam_dict.save_path}")

        # 3. saving predictions for best models: use NPY files
        assert osp.exists(self.pred_dict.save_path), "pred_dict doesn't exist"
        # save pred_dict file
        self.pred_dict.load()
        if verbose:
            print(f"Load results from {self.pred_dict.save_path}")

        # 4. saving pred_stats for best models: use NPY files
        # NOTE: is ok for the pred_stats to be not exist
        if osp.exists(self.pred_stats.save_path):
            # save self.pred_stats file
            self.pred_stats.load()
            if verbose:
                print(f"Load results from {self.pred_stats.save_path}")
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
                print(e)
                return
        elif self.task.task_type == 'classification':
            try:
                confusion_matrix_models(self.pred_dict,
                                        os.path.join(self.APP_DIR,
                                                     diagnosis_figure_file),
                                        classes=list(range(1, 6)))
            except Exception as e:
                print(e)
                return
        if verbose:
            print(f"Saved figure to {diagnosis_figure_file}")
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
        vis_heatmap(X['train'], save_path=os.path.join(
                    save_dir, f"{self.version}_heatmap_train.jpeg"),
                    xlabel='nodule', ylabel='features')
        vis_pca(data=X['train'], label=Y['train'],
                save_path=os.path.join(
                    save_dir, f"{self.version}_{self.task_name}_pca_train.jpeg"),
                label_name=self.task_name, **kwarg)
        vis_tsne(data=X['train'], label=Y['train'],
                 save_path=os.path.join(
                     save_dir, f"{self.version}_{self.task_name}_pca_train.jpeg"),
                 label_name=self.task_name, **kwarg)
        # val
        vis_heatmap(X['val'], save_path=os.path.join(
                    save_dir, f"{self.version}_heatmap_val.jpeg"))
        vis_pca(data=X['val'], label=Y['val'],
                save_path=os.path.join(
                    save_dir, f"{self.version}_{self.task_name}_pca_val.jpeg"),
                label_name=self.task_name, **kwarg)
        vis_tsne(data=X['val'], label=Y['val'],
                 save_path=os.path.join(
                     save_dir, f"{self.version}_{self.task_name}_pca_val.jpeg"),
                 label_name=self.task_name, **kwarg)
        pass
