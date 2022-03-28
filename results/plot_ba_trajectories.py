import yaml
from typing import List
import pandas as pd
from pathlib import Path
from scipy import stats

from ludwig.results import gen_param_paths

from treetransitions.figs import plot_ba_trajectories
from treetransitions.params import param2default, param2requests
from treetransitions import config
from treetransitions.utils import correct_artifacts

TOLERANCE = 0.05
PLOT_NUM_CATS_LIST = (32,)


def get_dfs(param_path: Path,
            name: str,
            ):

    # get csv file paths
    df_paths = list(param_path.glob(f'*num*/{name}.csv'))
    print(f'Found {len(df_paths)} results files')

    # load csv files
    dfs = []
    for df_path in df_paths:
        with df_path.open('rb') as f:
            try:
                df = correct_artifacts(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                print(f'{df_path.name} is empty. Skipping')
                return []
        dfs.append(df)

    return dfs


def to_dict(dfs: List[pd.DataFrame],
            sem: bool,
            ):


    concatenated = pd.concat(dfs, axis=0)
    grouped = concatenated.groupby(concatenated.index)
    df = grouped.mean() if not sem else grouped.agg(stats.sem)
    res = df.to_dict(orient='list')
    res = {int(k): v for k, v in res.items()}  # convert key to int - to be able to sort
    return res


# get results
project_name = config.Dirs.root.name
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         ):

    # load data
    dfs = get_dfs(param_path, 'num_cats2bas')

    # aggregate data
    num_dfs = len(dfs)
    print(f'Aggregating data from {num_dfs} files')
    num_cats2ba_means = to_dict(dfs, sem=False)
    num_cats2ba_sems = to_dict(dfs, sem=True)

    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)

    summary = (num_cats2ba_means, num_cats2ba_sems, param2val, num_dfs)

    # plot
    plot_ba_trajectories(summary,
                         PLOT_NUM_CATS_LIST,
                         )



