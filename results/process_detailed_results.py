import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from results.test_dicts import marker_styles, draw_order, metric_ylim, best_function, is_metric_increasing, \
    metric_short_name

excluded_methods = ['Simple split']

RESULTS_ROOT_DIR = 'detailed_results'
ROOT_DIR = 'std_out'
SCENARIO = ''
# RESULTS_ROOT_DIR = 'detailed_results_cc'
# ROOT_DIR = 'cc_out'
# SCENARIO = ' (CC)'

BOXPLOTS_ROOT_DIR = os.path.join(ROOT_DIR, 'boxplots')
BOXPLOTS_SVG_ROOT_DIR = os.path.join(BOXPLOTS_ROOT_DIR, 'svg')
DETAILED_PLOTS_ROOT_DIR = os.path.join(ROOT_DIR, 'detailed_plots')
DETAILED_PLOTS_SVG_ROOT_DIR = os.path.join(DETAILED_PLOTS_ROOT_DIR, 'svg')
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


def plot_metrics(metrics_df, oracle_metrics_df, marker_styles, draw_order):
    mean_metrics_df = metrics_df.groupby(['Dataset', 'ConstC', 'Method', 'c', 'Metric']) \
        .Value \
        .mean() \
        .reset_index(drop=False)
    mean_oracle_metrics_df = oracle_metrics_df.groupby(['Dataset', 'ConstC', 'Method', 'c', 'Metric']) \
        .Value \
        .mean() \
        .reset_index(drop=False)

    split_metric_dict = dict(tuple(mean_metrics_df.groupby('Metric')))
    for metric in split_metric_dict:
        print(f'Drawing {metric} plots')
        split_const_c_dict = dict(tuple(split_metric_dict[metric].groupby('ConstC')))

        for const_c in split_const_c_dict:
            split_dataset_dict = dict(tuple(split_const_c_dict[const_c].groupby('Dataset')))

            sns.set_theme()
            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            fig.subplots_adjust(hspace=0.4)

            const_c_string = f'znane c' if const_c else f'estymowane c'
            const_c_string_file = f'known_c' if const_c else f'estimated_c'

            for k, dataset_name in enumerate(split_dataset_dict):
                # k = k + 1  # missing Adult dataset

                i = k // 3
                j = k % 3
                ax = axs[i, j]

                dataset_df = split_dataset_dict[dataset_name]
                split_method_dict = dict(tuple(dataset_df.groupby('Method')))

                legend = []

                if metric == 'AUC':
                    method = 'Oracle'
                    legend.append(method)

                    method_df = mean_oracle_metrics_df.loc[
                        (mean_oracle_metrics_df.Metric == metric) & (mean_oracle_metrics_df.ConstC == const_c) &
                        (mean_oracle_metrics_df.Dataset == dataset_name)]
                    ax.plot(method_df.c, method_df.Value, **marker_styles[method])

                for method in draw_order:
                    if method not in split_method_dict:
                        continue
                    legend.append(method)

                    method_df = split_method_dict[method]
                    ax.plot(method_df.c, method_df.Value, **marker_styles[method])

                ax.legend(legend)
                ax.set_xlabel(r'Częstość etykietowania $c$')
                ax.set_ylabel(metric_short_name[metric])
                ax.set_ylim(*metric_ylim[metric])
                if metric in [r'Błąd estymacji częstości etykietowania',
                              r'Błąd estymacji prawdopodobieństwa a priori']:
                    title_string = f'{dataset_name}'
                else:
                    title_string = f'{dataset_name} ({const_c_string})'
                ax.set_title(title_string, fontsize=16, fontweight='bold')


            # for ax in axs.flat:
            #     ax.label_outer()

            if metric in [r'Błąd estymacji częstości etykietowania',
                          r'Błąd estymacji prawdopodobieństwa a priori']:
                suptitle_string = f'{metric}{SCENARIO}'
            else:
                suptitle_string = f'{metric} - {const_c_string}{SCENARIO}'
            plt.suptitle(suptitle_string, fontsize=20, fontweight='bold')

            plt.tight_layout(rect=[0, 0.01, 1, 0.98])
            plt.savefig(os.path.join(DETAILED_PLOTS_ROOT_DIR,
                                     f'{metric_short_name[metric]}_{const_c_string_file}.png'),
                        dpi=300,
                        bbox_inches='tight')
            plt.savefig(os.path.join(DETAILED_PLOTS_SVG_ROOT_DIR,
                                     f'{metric_short_name[metric]}_{const_c_string_file}.svg'),
                        bbox_inches='tight')
            # plt.show()
            plt.close()


def draw_boxplots(metrics_df):
    split_metric_dict = dict(tuple(metrics_df.groupby('Metric')))
    for metric in split_metric_dict:
        print(f'Drawing {metric} boxplots')
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
                    if metric in [r'Błąd estymacji częstości etykietowania',
                                  r'Błąd estymacji prawdopodobieństwa a priori']:
                        title_string = f'{dataset_name}, c = {c}'
                    else:
                        title_string = f'{dataset_name}, c = {c} ({const_c_string})'
                    ax.set_title(title_string, fontsize=16, fontweight='bold')
            # for ax in axs.flat:
            #     ax.label_outer()

            plt.suptitle(f'{metric}{SCENARIO}', fontsize=20, fontweight='bold')
            plt.tight_layout(rect=[0, 0.01, 1, 0.98])
            plt.savefig(os.path.join(BOXPLOTS_ROOT_DIR,
                                     f'{metric} - {const_c_string}.png'),
                        dpi=300,
                        bbox_inches='tight')
            plt.savefig(os.path.join(BOXPLOTS_SVG_ROOT_DIR,
                                     f'{metric} - {const_c_string}.svg'),
                        bbox_inches='tight')
            # plt.show()
            plt.close()


def get_latex_table(metric, pivot, mean_pivot, type, pivot_col):
    def prepare_pivot(pivot):
        if pivot_col in pivot.columns:
            pivot.loc[:, pivot_col] = pivot.loc[:, pivot_col].astype(str)

        cols_at_start = ['ConstC', 'index', pivot_col]
        cols_at_end = ['Oracle']

        cols = [c for c in pivot if c not in cols_at_start and c not in cols_at_end]
        cols.sort(key=lambda m: draw_order.index(m))

        pivot = pivot[[c for c in cols_at_start if c in pivot]
                      + cols
                      + [c for c in cols_at_end if c in pivot]]
        return pivot

    pivot = prepare_pivot(pivot)
    mean_pivot = prepare_pivot(mean_pivot)

    best = np.min
    if type == 'values':
        best = best_function[metric]

    for const_c in [True, False]:
        latex_string = '\\hline\n'
        latex_string += f"{pivot_col:16} "
        for col in pivot.columns:
            if col in ['ConstC', 'index', pivot_col]:
                continue
            if np.sum(np.isnan(pivot.loc[pivot.ConstC == const_c, col])):
                continue
            latex_string += f"& {col:14} "
        latex_string += '\\\\\n'
        latex_string += '\\hline\n'

        const_c_metric_pivot = pivot.loc[pivot.ConstC == const_c]

        for i, row in const_c_metric_pivot.iterrows():
            latex_string += f"{row[pivot_col]:<16} "

            values = [item[1] for (i, item) in enumerate(row.items())
                             if i >= 2 and not np.isnan(item[1])]
            non_oracle_values = [item[1] for (i, item) in enumerate(row.items())
                             if i >= 2 and not np.isnan(item[1]) and not item[0] == 'Oracle']

            for value in values:
                text = f"{value}"
                if value == best(non_oracle_values):
                    text = "\\textbf{" + text + "}"
                latex_string += f"& {text:14} "
            latex_string += '\\\\\n'

        if len(const_c_metric_pivot) > 0:
            latex_string += '\\hline\n'

            mean_row = mean_pivot.loc[mean_pivot.ConstC == const_c].iloc[0]
            latex_string += f"{'Średnia':16} "

            values = [item[1] for (i, item) in enumerate(mean_row.items())
                             if i >= 1 and not np.isnan(item[1])]
            non_oracle_values = [item[1] for (i, item) in enumerate(mean_row.items())
                             if i >= 1 and not np.isnan(item[1]) and not item[0] == 'Oracle']

            for value in values:
                text = f"{value}"
                if value == best(non_oracle_values):
                    text = "\\textbf{" + text + "}"
                latex_string += f"& {text:14} "
            latex_string += '\\\\\n'

        latex_string += '\\hline\n'

        with open(os.path.join(LATEX_ROOT_DIR,
                               f'{type}_by_{pivot_col}_{metric}_latex_{"known" if const_c else "estimated"}_c.txt'),
                  'w') as f:
            f.write(latex_string)


def create_rankings(metrics_df, oracle_metrics_df, pivot_col):  # pivot_col in ['Dataset', 'c']
    for metric in metrics_df.Metric.unique():
        print(f'Creating {metric} rankings')
        df = metrics_df.loc[metrics_df.Metric == metric]
        oracle_df = oracle_metrics_df.loc[oracle_metrics_df.Metric == metric]

        metric_values_df = df[['Dataset', 'ConstC', 'c', 'Method', 'Value']]
        oracle_metric_values_df = oracle_df[['Dataset', 'ConstC', 'c', 'Method', 'Value']]

        values_df = pd.concat([oracle_metric_values_df, metric_values_df])
        values_df.Value = values_df.Value.round(3)

        full_df = pd.concat([oracle_df, df])
        ranks = full_df.groupby(['ConstC', 'Dataset', 'c', 'RunNumber'])\
            .Value\
            .rank(ascending=is_metric_increasing[metric])
        ranked_metrics_df = full_df.assign(Rank=ranks)

        metric_pivot = pd.pivot_table(values_df, values='Value',
                                      index=['ConstC', pivot_col],
                                      columns=['Method']) \
            .reset_index()
        rank_pivot = pd.pivot_table(ranked_metrics_df, values='Rank',
                                    index=['ConstC', pivot_col],
                                    columns=['Method']) \
            .reset_index()

        mean_metric_pivot = pd.pivot_table(values_df, values='Value',
                                           index=['ConstC'],
                                           columns=['Method']) \
            .round(3) \
            .reset_index(drop=False)
        mean_rank_pivot = pd.pivot_table(ranked_metrics_df, values='Rank',
                                         index=['ConstC'],
                                         columns=['Method']) \
            .round(3) \
            .reset_index(drop=False)

        metric_pivot = metric_pivot.round(3)
        rank_pivot = rank_pivot.round(3)
        metric_pivot.to_csv(os.path.join(CSV_ROOT_DIR, f'metrics_pivot_{metric}_by_{pivot_col}.csv'))
        rank_pivot.to_csv(os.path.join(CSV_ROOT_DIR, f'ranks_pivot_{metric}_by_{pivot_col}.csv'))

        get_latex_table(metric, metric_pivot, mean_metric_pivot, type='values', pivot_col=pivot_col)
        get_latex_table(metric, rank_pivot, mean_rank_pivot, type='ranks', pivot_col=pivot_col)


if __name__ == '__main__':
    metrics_df, oracle_metrics_df = read_results(RESULTS_ROOT_DIR)
    oracle_metrics_df['RunNumber'] = oracle_metrics_df.groupby(['Metric', 'Dataset', 'ConstC', 'c']).cumcount() + 1

    metrics_df['c'] = np.round(metrics_df['c'], 1)
    oracle_metrics_df['c'] = np.round(oracle_metrics_df['c'], 1)

    metrics_df['Method'] = np.where((metrics_df.Metric == r'Błąd estymacji częstości etykietowania') &
                                    (metrics_df.Method == 'Weighted - EN'),
                                    'EN', metrics_df.Method)
    metrics_df['Method'] = np.where((metrics_df.Metric == r'Błąd estymacji częstości etykietowania') &
                                    (metrics_df.Method == 'Weighted - TIcE'),
                                    'TIcE', metrics_df.Method)
    metrics_df['Method'] = np.where((metrics_df.Metric == r'Błąd estymacji prawdopodobieństwa a priori') &
                                    (metrics_df.Method == 'Weighted - EN'),
                                    'EN', metrics_df.Method)
    metrics_df['Method'] = np.where((metrics_df.Metric == r'Błąd estymacji prawdopodobieństwa a priori') &
                                    (metrics_df.Method == 'Weighted - TIcE'),
                                    'TIcE', metrics_df.Method)
    metrics_df['Method'] = np.where(metrics_df.Method == 'Weighted - EN', 'Weighted (EN)', metrics_df.Method)
    metrics_df['Method'] = np.where(metrics_df.Method == 'Weighted - TIcE', 'Weighted (TIcE)', metrics_df.Method)

    marker_styles['Weighted (EN)'] = marker_styles['Weighted - EN']
    marker_styles['Weighted (TIcE)'] = marker_styles['Weighted - TIcE']
    marker_styles['EN'] = marker_styles['EN']
    marker_styles['TIcE'] = marker_styles['Weighted - TIcE']

    metrics_df = metrics_df.loc[~np.isin(metrics_df.Method, excluded_methods)]

    print(metrics_df)
    print(oracle_metrics_df)

    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)

    if not os.path.exists(CSV_ROOT_DIR):
        os.mkdir(CSV_ROOT_DIR)
    if not os.path.exists(LATEX_ROOT_DIR):
        os.mkdir(LATEX_ROOT_DIR)
    if not os.path.exists(DETAILED_PLOTS_ROOT_DIR):
        os.mkdir(DETAILED_PLOTS_ROOT_DIR)
    if not os.path.exists(DETAILED_PLOTS_SVG_ROOT_DIR):
        os.mkdir(DETAILED_PLOTS_SVG_ROOT_DIR)
    if not os.path.exists(BOXPLOTS_ROOT_DIR):
        os.mkdir(BOXPLOTS_ROOT_DIR)
    if not os.path.exists(BOXPLOTS_SVG_ROOT_DIR):
        os.mkdir(BOXPLOTS_SVG_ROOT_DIR)

    plot_metrics(metrics_df, oracle_metrics_df, marker_styles, draw_order)
    create_rankings(metrics_df, oracle_metrics_df, pivot_col='Dataset')
    create_rankings(metrics_df, oracle_metrics_df, pivot_col='c')
    # draw_boxplots(metrics_df)
