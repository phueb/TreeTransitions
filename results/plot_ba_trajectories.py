import numpy as np
from typing import Dict, Tuple
import pandas as pd
from pathlib import Path

from ludwig.results import gen_param_paths

from treetransitions.figs import plot_ba_trajectories
from treetransitions.params import param2default, param2requests
from treetransitions import config
from treetransitions.utils import correct_artifacts

TOLERANCE = 0.05


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
            ) -> Tuple[Dict[int, np.array],
                       Dict[int, np.array]]:
    """

    return two dicts. each has num_cats as key, and a list of statistics as values

    """
    num_cats2avg_ba = {}
    num_cats2std_ba = {}
    for col_name in set(concatenated.columns):
        print(col_name)
        mat = concatenated[col_name].values  # all columns with col_name
        num_cats2avg_ba[int(col_name[3:])] = mat.mean(axis=1)
        num_cats2std_ba[int(col_name[3:])] = mat.std(axis=1)

    return num_cats2avg_ba, num_cats2std_ba


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

    # statistics
    num_cats2avg_ba, num_cats2std_ba = to_dict(concatenated_df)
    summary = (num_cats2avg_ba, num_cats2std_ba, eval_steps, num_dfs)

    # plot
    plot_ba_trajectories(summary,
                         label,
                         )



