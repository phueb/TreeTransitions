import yaml
import pandas as pd
import numpy as np
from scipy import stats

from ludwig.results import gen_param_paths

from treetransitions.figs import plot_ba_trajectories
from treetransitions.params import param2default, param2requests
from treetransitions import config


TOLERANCE = 0.05
PLOT_NUM_CATS_LIST = (32,)


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
project_name = config.Dirs.root.name
for param_p, label in gen_param_paths(project_name,
                                      param2requests,
                                      param2default,
                                      ):
    bas_df = get_dfs(param_p, 'num_cats2bas')
    num_cats2ba_means = to_dict(bas_df, sem=False)
    num_cats2ba_sems = to_dict(bas_df, sem=True)
    num_results = len(bas_df)

    with (param_p / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)

    summary = (num_cats2ba_means, num_cats2ba_sems, param2val, num_results)

    plot_ba_trajectories(summary,
                         PLOT_NUM_CATS_LIST,
                         )



