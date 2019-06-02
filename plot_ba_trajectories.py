import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats

from ludwigcluster.utils import list_all_param2vals
from treetransitions import config
from treetransitions.params import Params as MatchParams


VERBOSE = True

Y_MAX = 1.0
X_STEP = 5
GRID = False
PLOT_MAX_BA = False
LEGEND = True
YLIMs = None
TITLE_FONTSIZE = 10
PLOT_NUM_CATS_LIST = [2, 4, 8, 16, 32]
TOLERANCE = 0.05
PLOT_COMPARISON = True
CONFIDENCE = 0.95


default_dict = MatchParams.__dict__.copy()
# MatchParams.legal_probs = [[1.0, 1.0], [0.5, 1.0]]


def gen_param_ps(param2requested, param2default):
    compare_params = [param for param, val in param2requested.__dict__.items()
                      if val != param2default[param]]
    for param_p in config.RemoteDirs.runs.glob('param_*'):
        print('Checking {}...'.format(param_p))
        with (param_p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        param2val = param2val.copy()
        match_param2vals = list_all_param2vals(param2requested, add_names=False)
        del param2val['param_name']
        del param2val['job_name']
        if param2val in match_param2vals:
            print('Param2val matches')
            label = '\n'.join(['{}={}'.format(param, param2val[param]) for param in compare_params])
            yield param_p, label


def make_title(param2val):
    res = ''
    for param, val in sorted(param2val.items(), key=lambda i: i[0]):
        res += '{}={}\n'.format(param, val)
    return res


def make_comparison_title(param2val1, param2val2):
    res = 'Comparison between '
    for param, val1 in param2val1.items():
        if param == 'param_name':
            continue
        val2 = param2val2[param]
        if val1 != val2:
            res += '{}={} vs. {} (dashed line)\n'.format(param, val1, val2)
    return res


def correct_artifacts(df):
    # correct for ba algorithm - it results in negative spikes occasionally
    for bas in df.values.T:
        num_bas = len(bas)
        for i in range(num_bas - 2):
            val1, val2, val3 = bas[[i, i+1, i+2]]
            if (val1 - TOLERANCE) > val2 < (val3 - TOLERANCE):
                bas[i+1] = np.mean([val1, val3])
    return df


def get_dfs(param_p, name):
    results_ps = list(param_p.glob('*num*/{}.csv'.format(name)))
    print('Found {} results files'.format(len(results_ps)))
    dfs = []
    for results_p in results_ps:
        with results_p.open('rb') as f:
            try:
                df = correct_artifacts(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                print('{} is empty. Skipping'.format(results_p.name))
                return []
        dfs.append(df)
    return dfs


def to_dict(dfs, sem):
    print('Combining results from {} files'.format(len(dfs)))
    concatenated = pd.concat(dfs, axis=0)
    grouped = concatenated.groupby(concatenated.index)
    df = grouped.mean() if not sem else grouped.agg(stats.sem)
    res = df.to_dict(orient='list')
    res = {int(k): v for k, v in res.items()}  # convert key to int - to be able to sort
    return res


def plot_ba_trajs(results):
    # read results
    if len(results) == 1:
        num_cats2ba_means1, num_cats2ba_sems1, num_cats2max_ba1, param2val1, num_results1 = results[0]
        title = make_title(param2val1) + '\nn={}'.format(num_results1)
        d1s = [num_cats2ba_means1]
        d2s = [num_cats2ba_sems1]
        d3s = [num_cats2max_ba1]
        dofs = [num_results1 - 1]
        figsize = (10, 10)
    elif len(results) == 2:
        num_cats2ba_means1, num_cats2ba_sems1, num_cats2max_ba1, param2val1, num_results1 = results[0]
        num_cats2ba_means2, num_cats2ba_sems2, num_cats2max_ba2, param2val2, num_results2 = results[1]
        title = make_comparison_title(param2val1, param2val2) + '\nn={}'.format(min(num_results1, num_results2))
        assert param2val1['num_partitions'] == param2val2['num_partitions']
        assert param2val1['num_iterations'] == param2val2['num_iterations']
        d1s = [num_cats2ba_means1, num_cats2ba_means2]
        d2s = [num_cats2ba_sems1, num_cats2ba_sems2]
        d3s = [num_cats2max_ba1, num_cats2max_ba2]
        dofs = [num_results1 - 1, num_results2 - 1]
        figsize = (10, 8)
    else:
        raise ValueError('"results" cannot contain more than 2 entries.')
    # fig
    fig, ax = plt.subplots(figsize=figsize, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Balanced Accuracy +/- {}%-Confidence Interval'.format(CONFIDENCE * 100))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ticks (ba is calculated before training update, so first ba is x=0, and last x=num_ba-1)
    num_x = param2val1['num_partitions'] * param2val1['num_iterations']
    xticks = np.arange(0, num_x + X_STEP, X_STEP)
    yticks = np.linspace(0.5, 1.00, 6, endpoint=True).round(2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim([0.5, Y_MAX])
    ax.set_xlim([0, xticks[-1]])
    if GRID:
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
    # plot
    x = np.arange(num_x)
    for n, (d1, d2, d3, dof) in enumerate(zip(d1s, d2s, d3s, dofs)):
        num_trajs = len(d1)
        palette = iter(sns.color_palette('hls', num_trajs))
        for num_cats in PLOT_NUM_CATS_LIST:
            c = next(palette)
            # ba mean
            ba_means = np.asarray(d1[num_cats])
            ax.plot(x, ba_means, color=c,
                    label='num_cats={}'.format(num_cats) if n == 0 else '_nolegend_',
                    linestyle='-' if n == 0 else ':')

            # TODO test conf int
            # dof = 20

            # ba conf_int
            ba_sems = np.asarray(d2[num_cats])
            q = (1 - CONFIDENCE) / 2.
            margins = ba_sems * stats.t.ppf(q, dof)
            ax.fill_between(x, ba_means - margins, ba_means + margins, color=c, alpha=0.2)
            # max_ba
            if d3 is not None and PLOT_MAX_BA:
                ax.axhline(y=d3[num_cats], linestyle='dashed', color=c, alpha=0.5)
    if LEGEND:
        plt.legend(loc='best', frameon=False)
    plt.show()


def gen_results_from_disk():
    for param_p, label in gen_param_ps(MatchParams, default_dict):
        bas_df = get_dfs(param_p, 'num_cats2bas')
        max_ba_dfs = get_dfs(param_p, 'num_cats2max_ba')
        num_cats2ba_means = to_dict(bas_df, sem=False)
        num_cats2ba_sems = to_dict(bas_df, sem=True)
        num_cats2max_ba = to_dict(max_ba_dfs, sem=False) if max_ba_dfs else None
        num_results = len(bas_df)
        #
        with (param_p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)

        print(param2val['legal_probs'])

        yield (num_cats2ba_means, num_cats2ba_sems, num_cats2max_ba, param2val, num_results)


if __name__ == '__main__':
    all_results = list(gen_results_from_disk())

    if not PLOT_COMPARISON:
        # plot each single result
        for single_result in all_results:
            plot_ba_trajs([single_result])
    else:
        # plot each result pair
        for results_pair in combinations(all_results, 2):
            plot_ba_trajs(results_pair)



