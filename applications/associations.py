# functions related to association analysis
from tqdm import tqdm
import pandas as pd


def get_stats_results(X, Y, func):
    results = []
    for col in tqdm(range(X.shape[1])):
        results.append(list(func(X[:, col], Y)))
    return pd.DataFrame(results, columns=['correlation', 'pvalue'])
