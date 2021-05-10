import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from results.test_dicts import marker_styles, draw_order, metric_ylim, best_function, is_metric_increasing

RESULTS_ROOT_DIR = 'detailed_results'
ROOT_DIR = 'std_out'
SCENARIO = ''
# RESULTS_ROOT_DIR = 'detailed_results_cc'
# ROOT_DIR = 'cc_out'
# SCENARIO = ' (CC)'

BOXPLOTS_ROOT_DIR = os.path.join(ROOT_DIR, 'boxplots')
DETAILED_PLOTS_ROOT_DIR = os.path.join(ROOT_DIR, 'detailed_plots')
LATEX_ROOT_DIR = os.path.join(ROOT_DIR, 'latex')
CSV_ROOT_DIR = os.path.join(ROOT_DIR, 'csv')
# PLOTS_ROOT_DIR = os.path.join(ROOT_DIR, 'plots')


def read_results(root_dir):
    oracle_dfs = []
    metrics_dfs = []

    for dataset in os.listdir(root_dir):
        for file in os.listdir(os.path.join(root_dir, dataset)):
            if os.path.isdir(os.path.join(root_dir, dataset, file)):  # oracle metrics
                for oracle_file in os.listdir(os.path.join(root_dir, dataset, file)):
                    df = pd.read_csv(os.path.join(root_dir, dataset, file, oracle_file))
                    oracle_dfs.append(df)
            else:  # method metrics
                df = pd.read_csv(os.path.join(root_dir, dataset, file))
                metrics_dfs.append(df)

    oracle_df = pd.concat(oracle_dfs)
    metrics_df = pd.concat(metrics_dfs)
    return metrics_df, oracle_df


def plot_metrics(metrics_df, marker_styles, draw_order):
    mean_metrics_df = metrics_df.groupby(['Dataset', 'ConstC', 'Method', 'c', 'Metric']) \
        .Value \
        .mean() \
        .reset_index(drop=False)

    split_metric_dict = dict(tuple(mean_metrics_df.groupby('Metric')))
    for metric in split_metric_dict:
        split_const_c_dict = dict(tuple(split_metric_dict[metric].groupby('ConstC')))

        for const_c in split_const_c_dict:
            split_dataset_dict = dict(tuple(split_const_c_dict[const_c].groupby('Dataset')))

            sns.set_theme()
            fig, axs = plt.subplots(3, 3, figsize=(18, 18))
            fig.subplots_adjust(hspace=0.4)

            const_c_string = f'znane c' if const_c else f'estymowane c'

            for k, dataset_name in enumerate(split_dataset_dict):
                # k = k + 1  # missing Adult dataset

                i = k // 3
                j = k % 3
                ax = axs[i, j]

                dataset_df = split_dataset_dict[dataset_name]
                split_method_dict = dict(tuple(dataset_df.groupby('Method')))

                legend = []
                for method in draw_order:
                    if method not in split_method_dict:
                        continue
                    legend.append(method)

                    method_df = split_method_dict[method]
                    ax.plot(method_df.c, method_df.Value, **marker_styles[method])

                ax.legend(legend)
                ax.set_xlabel(r'Częstość etykietowania $c$')
                ax.set_ylabel(metric)
                ax.set_ylim(*metric_ylim[metric])
                ax.set_title(f'{dataset_name} ({const_c_string})', fontsize=16, fontweight='bold')

            # for ax in axs.flat:
            #     ax.label_outer()

            plt.suptitle(f'{metric}{SCENARIO}', fontsize=20, fontweight='bold')
            plt.savefig(os.path.join(DETAILED_PLOTS_ROOT_DIR,
                                     f'{metric} - {const_c_string}.png'),
                        dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()


def draw_boxplots(metrics_df):
    split_metric_dict = dict(tuple(metrics_df.groupby('Metric')))
    for metric in split_metric_dict:
        split_const_c_dict = dict(tuple(split_metric_dict[metric].groupby('ConstC')))

        for const_c in split_const_c_dict:
            split_dataset_dict = dict(tuple(split_const_c_dict[const_c].groupby('Dataset')))

            const_c_string = f'znane c' if const_c else f'estymowane c'

            all_datasets = split_const_c_dict[const_c].Dataset.unique()
            all_cs = split_const_c_dict[const_c].c.unique()

            sns.set_theme()
            fig, axs = plt.subplots(len(all_datasets), len(all_cs), figsize=(6 * len(all_cs), 6 * len(all_datasets)))
            fig.subplots_adjust(hspace=0.4)

            for i, dataset_name in enumerate(split_dataset_dict):

                dataset_df = split_dataset_dict[dataset_name]
                split_c_dict = dict(tuple(dataset_df.groupby('c')))

                for j, c in enumerate(split_c_dict):
                    ax = axs[i, j]

                    c_df = split_c_dict[c]
                    sns.boxplot(x='Method', y='Value', ax=ax, data=c_df)

                    ax.set_xlabel(r'Metoda')
                    ax.set_ylabel(metric)
                    ax.set_ylim(*metric_ylim[metric])
                    ax.set_title(f'{dataset_name}, c = {c} ({const_c_string})', fontsize=16, fontweight='bold')

            # for ax in axs.flat:
            #     ax.label_outer()

            plt.suptitle(f'{metric}{SCENARIO}', fontsize=20, fontweight='bold')
            plt.savefig(os.path.join(BOXPLOTS_ROOT_DIR,
                                     f'{metric} - {const_c_string}.png'),
                        dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()


def get_latex_table(metric, metric_pivot, rank_pivot):
    metric_pivot = metric_pivot.reset_index()
    rank_pivot = rank_pivot.reset_index()

    for const_c in [True, False]:
        latex_string = f"{'Dataset':16} "
        for col in metric_pivot.columns:
            if col in ['ConstC', 'index', 'Dataset']:
                continue
            if np.sum(np.isnan(metric_pivot.loc[metric_pivot.ConstC == const_c, col])):
                continue
            latex_string += f"& {col:14} "
        latex_string += '\\\\\n'

        for row in metric_pivot.loc[metric_pivot.ConstC == const_c].iterrows():
            latex_string += f"{row[1]['Dataset']:16} "

            values = [item[1] for (i, item) in enumerate(row[1].items())
                             if i >= 2 and not np.isnan(item[1])]
            non_oracle_values = [item[1] for (i, item) in enumerate(row[1].items())
                             if i >= 2 and not np.isnan(item[1]) and not item[0] == 'Oracle']

            for value in values:
                text = f"{value}"
                if value == best_function[metric](non_oracle_values):
                    text = "\\textbf{" + text + "}"
                latex_string += f"& {text:14} "
            latex_string += '\\\\\n'

        for row in rank_pivot.loc[rank_pivot.ConstC == const_c].iterrows():
            latex_string += f"{'Rank':16} "
            values = [item[1] for (i, item) in enumerate(row[1].items())
                      if i >= 1 and not np.isnan(item[1])]
            non_oracle_values = [item[1] for (i, item) in enumerate(row[1].items())
                      if i >= 1 and not np.isnan(item[1]) and not item[0] == 'Oracle']
            for value in values:
                text = f"{value}"
                if value == np.min(non_oracle_values):
                    text = "\\textbf{" + text + "}"
                latex_string += f"& {text:14} "
            latex_string += '\\\\\n'

        with open(os.path.join(LATEX_ROOT_DIR,
                               f'{metric}_latex_{"known" if const_c else "estimated"}_c.txt'),
                  'w') as f:
            f.write(latex_string)


def create_rankings(metrics_df, oracle_metrics_df):
    for metric in metrics_df.Metric.unique():
        df = metrics_df.loc[metrics_df.Metric == metric]
        oracle_df = oracle_metrics_df.loc[oracle_metrics_df.Metric == metric]

        mean_metrics_df = df.groupby(['Dataset', 'ConstC', 'Method']) \
            .Value \
            .mean() \
            .reset_index(drop=False)
        mean_oracle_metrics_df = oracle_df.groupby(['Dataset', 'ConstC', 'Method']) \
            .Value \
            .mean() \
            .reset_index(drop=False)

        mean_metrics_df = pd.concat([mean_oracle_metrics_df, mean_metrics_df])
        mean_metrics_df.Value = mean_metrics_df.Value.round(3)

        ranks = mean_metrics_df.groupby(['Dataset', 'ConstC'])\
            .Value\
            .rank(ascending=is_metric_increasing[metric])
        ranked_mean_metrics_df = mean_metrics_df.assign(Rank=ranks)
        mean_rank = ranked_mean_metrics_df\
            .groupby(['ConstC', 'Method'])\
            .Rank\
            .mean()\
            .reset_index()

        metric_pivot = pd.pivot_table(mean_metrics_df, values='Value',
                                      index=['ConstC', 'Dataset'],
                                      columns=['Method'])\
            .round(3)
        rank_pivot = pd.pivot_table(mean_rank, values='Rank',
                                    index=['ConstC'], columns=['Method'])\
            .round(3)

        metric_pivot.to_csv(os.path.join(CSV_ROOT_DIR, f'mean_metrics_by_dataset_{metric}.csv'))
        rank_pivot.to_csv(os.path.join(CSV_ROOT_DIR, f'mean_ranks_{metric}.csv'))

        get_latex_table(metric, metric_pivot, rank_pivot)


if __name__ == '__main__':
    metrics_df, oracle_df = read_results(RESULTS_ROOT_DIR)

    print(metrics_df)
    print(oracle_df)

    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)

    if not os.path.exists(BOXPLOTS_ROOT_DIR):
        os.mkdir(BOXPLOTS_ROOT_DIR)
    if not os.path.exists(CSV_ROOT_DIR):
        os.mkdir(CSV_ROOT_DIR)
    if not os.path.exists(LATEX_ROOT_DIR):
        os.mkdir(LATEX_ROOT_DIR)
    if not os.path.exists(DETAILED_PLOTS_ROOT_DIR):
        os.mkdir(DETAILED_PLOTS_ROOT_DIR)

    plot_metrics(metrics_df, marker_styles, draw_order)
    create_rankings(metrics_df, oracle_df)
    draw_boxplots(metrics_df)
