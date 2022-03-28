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
PLOT_NUM_CATS_LIST = (2, 4, 8, 16, 32,)


def get_series_list(param_path: Path,
                    pattern: str,
                    ):

    # get csv file paths
    df_paths = list(param_path.glob(f'*num*/{pattern}.csv'))

    if not df_paths:
        raise RuntimeError(f'Did not find csv files in {param_path}')
    else:
        print(f'Found {len(df_paths)} csv files')

    # load csv files
    dfs = []
    for df_path in df_paths:
        with df_path.open('rb') as f:
            try:
                df = pd.read_csv(f, index_col=0, squeeze=True)
                # df = correct_artifacts(df, TOLERANCE)  # TODO fix
            except pd.errors.EmptyDataError:
                print(f'{df_path.name} is empty. Skipping')
                return []
        dfs.append(df)

    return dfs


def to_dict(concatenated: pd.DataFrame,
            do_sem: bool,
            ):

    grouped = concatenated.groupby(concatenated.index)
    df = grouped.mean() if not do_sem else grouped.agg(stats.sem)
    res = df.to_dict(orient='list')
    res = {int(k[3:]): v for k, v in res.items()}  # convert key to float - to be able to sort

    return res


# get results
project_name = config.Dirs.root.name
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         ):

    # load data
    series_list = get_series_list(param_path, 'ba_*')

    # aggregate data
    num_dfs = len(series_list)
    print(f'Aggregating data from {num_dfs} files')
    concatenated_df = pd.concat(series_list, axis=1)
    eval_steps = concatenated_df.index.values

    print(concatenated_df)

    # statistics
    num_cats2ba_means = to_dict(concatenated_df, do_sem=False)
    num_cats2ba_sems = to_dict(concatenated_df, do_sem=True)
    summary = (num_cats2ba_means, num_cats2ba_sems, eval_steps, num_dfs)

    # with (param_path / 'param2val.yaml').open('r') as f:
    #     param2val = yaml.load(f, Loader=yaml.FullLoader)

    # plot
    plot_ba_trajectories(summary,
                         label,
                         PLOT_NUM_CATS_LIST,
                         )



