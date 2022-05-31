import os.path as osp
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def organize(versions=["VAE3D32AUG_49", "VAE3D32AUG_51", "VAE3D32AUG_53",
                       "PRETRAINED_VAE_1", "PRETRAINED_VAE_2", "PRETRAINED_VAE_3",
                       "VAE3D32AUG_60"],
             model_names=None,
             task_name=None,
             metric="AUROC", task="StfAJCC", model="xgboost", save_name="boxplot:cv"):
    result_dict = summarize_mul_models(
        metric=metric, task=task, model=model, versions=versions,)
    result_df = pd.DataFrame(result_dict)
    if not model_names:
        model_names = [f"model{i}" for i in range(1, len(versions)+1)]
    result_df.columns = model_names
    plot_box(result_df,  xlabel="Models",
             ylabel=f"F1 score - {task_name}" if task_name else "F1 score",
             ylim=True,
             figsize=(8, 5),
             save_path=osp.join("/labs/gevaertlab/users/yyhhli/code/vae/EDA/downstream_tasks",
                                f"vis_results/{save_name}_{model}_{metric}_{task}.jpeg"))
    pass


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def summarize_metrics(result_dict, metric_list=["Accuracy", "F1", "Precision", "Recall", "AUROC"]):
    metrics = {m: [] for m in metric_list}
    for k, rdict in result_dict.items():
        if k != "__dict":
            for m, v in rdict.items():
                metrics[m].append(v)
    return metrics


def save_results(label_names=["StfAJCC", "StfEGFRMutation", "StfLymphInvasion", "StfNStage", "StfTStage"], model_version_list=[49, 51, 53, 57, 58, 59, 60],
                 versions=["VAE3D32AUG_60", "VAE3D32AUG_49",
                           "VAE3D32AUG_51", "VAE3D32AUG_53"],
                 ):
    # prepare data for table making, have to put all the labels and model versions here
    for label in label_names:
        result = {}
        for ver in versions:
            data = load_json(osp.join(ver, label+".result_dict.json"))
            sdata = summarize_metrics(data)
            result[ver] = [np.mean(sdata[n]) for n in sdata.keys()]
        df = pd.DataFrame(result)
        df.to_csv(f"{label}.csv", index=False)
    pass


def summarize_mul_models(metric="AUROC", task="StfAJCC", model="random_forest",
                         versions=["VAE3D32AUG_60", "VAE3D32AUG_49", "VAE3D32AUG_51", "VAE3D32AUG_53"]):
    # prepare data for visualization
    result_root = "/labs/gevaertlab/users/yyhhli/code/vae/applications/results"
    result_dict = {}
    for ver in versions:
        data = load_json(osp.join(result_root, ver, task +
                         ".cv_result_dict.json"))
        result_dict[ver] = data[model][metric]
    return result_dict


def plot_box(data, save_path=None, xlabel=None, ylabel=None, ylim=True, figsize=(8, 5)):
    # data: pandas dataframe cols=model_names, rows=metrics
    plt.figure(figsize=figsize)
    if ylim is True:
        ylim = (0, 1)
    if ylim:
        plt.ylim(ylim)
    ##### Set style options here #####
    sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
    boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
    flierprops = dict(marker='o', markersize=1,
                      linestyle='none')
    whiskerprops = dict(color='#00145A')
    capprops = dict(color='#00145A')
    medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')
    meanprops = dict(marker='D', markeredgecolor='#fb0172')
    vals, names, xs = [], [], []
    for i, col in enumerate(data.columns):
        vals.append(data[col].values)
        names.append(col)
        xs.append(np.random.normal(i, 0.04, data[col].values.shape[0]))
        # adds jitter to the data points - can be adjusted
    plt.boxplot(data, notch=False,  # labels=data.columns,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, flierprops=flierprops,
                medianprops=medianprops, showmeans=True, meanprops=meanprops)
    palette = ['#FF2709', '#09FF10', '#0030D7', '#FA70B5',
               "#21325E", "#EF6D6D", "#573391", "#00B8A9"]
    for x, val, c in zip(xs, vals, palette[:len(vals)]):
        plt.scatter(x, val, alpha=0.4, color=c)
    # add titles
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=400)
        print(f"saved to {save_path}")
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Draw boxplots for downstream tasks comparisons')

    parser.add_argument("--model_names", "-n", nargs="+", default="")
    parser.add_argument("--versions", "-v", nargs="+",
                        default=["VAE3D32AUG_49", "VAE3D32AUG_51", "VAE3D32AUG_53",
                                 "PRETRAINED_VAE_4", "PRETRAINED_VAE_5", "PRETRAINED_VAE_6", "VAE3D32AUG_60",
                                 ])  # VAE3D32AUG_70

    parser.add_argument("--metric", "-m", default="F1")
    parser.add_argument("--tasks", "-t", nargs="+", default=["StfAJCC", "StfEGFRMutation", "StfHisGrade",
                        "StfKRASMutation", "StfLymphInvasion", "StfNStage", "StfTStage", "StfPleuralInvasion"])
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--task_names", nargs="+", default=["AJCC Score", "EGFR Mutation", "Histopathologic Grade",
                        "KRAS Mutation", "Lymphovascular Invasion ", "Pathological N Stage", "Pathological T Stage", "Pleural Invasion"])  # printed names for tasks
    parser.add_argument("--save_name", default="boxplot:cv")
    args = parser.parse_args()
    for i, task in enumerate(args.tasks):
        organize(versions=args.versions,
                 metric=args.metric,
                 task_name=args.task_names[i],
                 task=task,
                 model=args.model,
                 model_names=args.model_names,
                 save_name=args.save_name)
    pass

# combined training: "VAE3D32AUG_49" "VAE3D32AUG_70" "VAE3D32AUG_51" "VAE3D32AUG_60"
# fine-tuning: VAE3D32AUG_49 VAE3D32AUG_70 VAE3D32AUG_53 PRETRAINED_VAE_6 VAE3D32AUG_60
