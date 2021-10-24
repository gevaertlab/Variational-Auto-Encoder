# perform association analysis of embeddings and labels


from typing import Callable, List
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

from tqdm import tqdm
from evaluations.export import Exporter
from scipy.stats import spearmanr


class AssociationAnalysis(Exporter):

    def __init__(self,
                 base_model_name: str,
                 log_name: str,
                 version: int,
                 task_names: List[str]):
        super().__init__(base_model_name,
                         log_name,
                         version,
                         task_names=task_names)
        self.embeddings, self.data_names, self.label_dict = self.get_data()
        pass

    def association(self, corr_func: Callable, correction_func: Callable):
        sig_dict = {}
        for task_name in self.task_names:
            self.logger.info(f"---start task {task_name}---")
            X, Y = self.embeddings['train'], self.label_dict[task_name]['train']
            self.logger.info(X['train'].shape, len(Y['train']))
            stats_df = self.get_stats_results(
                X['train'], Y['train'], corr_func)
            # correction
            fdr_df = pd.DataFrame(correction_func(
                pvals=list(stats_df['pvalue']))).transpose()
            fdr_df.columns = ['reject', 'adjusted']
            adj_stats_df = pd.concat([stats_df, fdr_df], axis=1)
            sig_df = adj_stats_df[adj_stats_df['reject'] == True]
            sig_dict[task_name] = sig_df
        return sig_dict

    def get_stats_results(self, X, Y, corr_func: Callable):
        results = []
        for col in tqdm(range(X.shape[1])):
            results.append(list(corr_func(X[:, col], Y)))
        return pd.DataFrame(results, columns=['correlation', 'pvalue'])

    def __call__(self):
        # use spearman's r and false discovery rate correction
        return self.association(corr_func=spearmanr, correction_func=fdrcorrection())
