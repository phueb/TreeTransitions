import yaml
import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats

from ludwig.results import gen_param_paths

from treetransitions.figs import plot_ba_trajectories
from treetransitions.params import param2default, param2requests
from treetransitions import config

VERBOSE = True

SOLID_LINE_ID = 0
YLIMs = None
TOLERANCE = 0.05
PLOT_COMPARISON = True
PLOT_NUM_CATS_LIST = (32,)

# CUSTOM_COMPARISON_TITLE = None # r'$C_3$ (dashed line) vs. $C_3reducedComplexity$ (solid line)'  # or None
CUSTOM_COMPARISON_TITLE = 'The effect of training order on semantic categorization\n$C_3startingSmall$'  # or None


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


# get results
results_list = []
project_name = config.Dirs.root.name
for param_p, label in gen_param_paths(project_name, param2requests, param2default):
    bas_df = get_dfs(param_p, 'num_cats2bas')
    max_ba_dfs = get_dfs(param_p, 'num_cats2max_ba')
    num_cats2ba_means = to_dict(bas_df, sem=False)
    num_cats2ba_sems = to_dict(bas_df, sem=True)
    num_cats2max_ba = to_dict(max_ba_dfs, sem=False) if max_ba_dfs else None
    num_results = len(bas_df)

    with (param_p / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)

    res = (num_cats2ba_means, num_cats2ba_sems, num_cats2max_ba, param2val, num_results)
    results_list.append(res)

if not PLOT_COMPARISON:
    # plot each single result
    for single_result in results_list:
        plot_ba_trajectories((single_result, ), PLOT_NUM_CATS_LIST)
else:
    # plot each result pair
    for results_pair in combinations(results_list, 2):
        plot_ba_trajectories(results_pair, PLOT_NUM_CATS_LIST, title=CUSTOM_COMPARISON_TITLE)



