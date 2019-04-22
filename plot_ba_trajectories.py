import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ludwigcluster.utils import list_all_param2vals
from treetransitions import config
from treetransitions.params import DefaultParams as MatchParams


VERBOSE = True

YLIMs = None
FIGSIZE = (10, 10)
TITLE_FONTSIZE = 10


default_dict = MatchParams.__dict__.copy()
MatchParams.num_levels = [10]
MatchParams.num_hiddens = [128]
MatchParams.parent_count = [1024]
MatchParams.num_partitions = [2]


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


def make_title(param_p):
    with (param_p / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    #
    res = ''
    for param, val in sorted(param2val.items(), key=lambda i: i[0]):
        res += '{}={}\n'.format(param, val)
    return res


def get_results_dfs(param_p):
    results_ps = list(param_p.glob('*num*/results.csv'))
    print('Found {} results files'.format(len(results_ps)))
    dfs = []
    for results_p in results_ps:
        with results_p.open('rb') as f:
            df = pd.read_csv(f)
        dfs.append(df)
    return dfs


def make_num_cats2bas(dfs):
    print('Combining results from {} files'.format(len(dfs)))
    concatenated = pd.concat(dfs, axis=0)
    df = concatenated.groupby(concatenated.index).mean()
    res = df.to_dict(orient='list')
    res = {int(k): v for k, v in res.items()}  # convert key to int - to be able to sort
    return res


def plot_ba_trajs(d1, d2, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Balanced Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0.5, 1.0])
    # plot
    num_trajs = len(d1)
    palette = iter(sns.color_palette('hls', num_trajs))
    for num_cats, bas in sorted(d1.items(), key=lambda i: i[0]):
        c = next(palette)
        ax.plot(bas, '-', color=c,
                label='num_cats={}'.format(num_cats))
        if d2 is not None:
            ax.axhline(y=d2[num_cats], linestyle='dashed', color=c)
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()


# plot
summary_data = []
for param_p, label in gen_param_ps(MatchParams, default_dict):
    results_dfs = get_results_dfs(param_p)
    num_cats2bas = make_num_cats2bas(results_dfs)
    #
    num_cats2max_ba = None  # TODO
    title = make_title(param_p)
    plot_ba_trajs(num_cats2bas, num_cats2max_ba, title + '\nn={}'.format(len(results_dfs)))