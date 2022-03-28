import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from typing import Optional, Tuple, List


def plot_heatmap(mat: np.array,
                 y_tick_labels: List[str],
                 x_tick_labels: List[str],
                 title: str = '',
                 x_label: str = '',
                 tick_label_font_size=1,
                 fontsize=16,
                 ):

    fig, ax = plt.subplots(figsize=(8, 8), dpi=None)
    plt.title(title, fontsize=fontsize)

    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap='jet',
              interpolation='nearest')
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel('Context words', fontsize=fontsize)

    # ticks
    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(x_tick_labels, rotation=90, fontsize=tick_label_font_size)
    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(y_tick_labels,  # no need to reverse (because no extent is set)
                            rotation=0, fontsize=tick_label_font_size)

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    plt.show()


def plot_ba_trajectories(results: Tuple[Tuple[dict, dict, dict, dict, int], ...],
                         plot_num_cats_list: Tuple[int],
                         y_max: float = 1.0,
                         x_step: int = 5,
                         is_grid: bool = False,
                         plot_max_ba: bool = False,
                         confidence: float = 0.95,
                         solid_line_id: int = 0,  # 0 or 1 to flip which line is solid
                         legend: bool = True,
                         title: Optional[str] = None,
                         fontsize=16):

    # read results
    if len(results) == 1:
        num_cats2ba_means1, num_cats2ba_sems1, num_cats2max_ba1, param2val1, num_results1 = results[0]
        if title is None:
            title_base = ''
            for param, val in sorted(param2val1.items(), key=lambda i: i[0]):
                title_base += '{}={}\n'.format(param, val)
            title = title_base + '\nn={}'.format(num_results1)
        d1s = [num_cats2ba_means1]
        d2s = [num_cats2ba_sems1]
        d3s = [num_cats2max_ba1]
        dofs = [num_results1 - 1]
        figsize = (8, 8)
    elif len(results) == 2:
        num_cats2ba_means1, num_cats2ba_sems1, num_cats2max_ba1, param2val1, num_results1 = results[0]
        num_cats2ba_means2, num_cats2ba_sems2, num_cats2max_ba2, param2val2, num_results2 = results[1]
        if title is None:
            title = make_comparison_title(param2val1, param2val2, solid_line_id)
            title += '\nn={}'.format(min(num_results1, num_results2))
        assert param2val1['num_partitions'] == param2val2['num_partitions']
        assert param2val1['num_iterations'] == param2val2['num_iterations']
        d1s = [num_cats2ba_means1, num_cats2ba_means2]
        d2s = [num_cats2ba_sems1, num_cats2ba_sems2]
        d3s = [num_cats2max_ba1, num_cats2max_ba2]
        dofs = [num_results1 - 1, num_results2 - 1]
        figsize = (8, 6)
    else:
        raise ValueError('"results" cannot contain more than 2 entries.')

    # fig
    fig, ax = plt.subplots(figsize=figsize, dpi=None)
    plt.title(title, fontsize=fontsize)
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Balanced Accuracy +/-\n{}%-Confidence Interval'.format(confidence * 100), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ticks (ba is calculated before training update, so first ba is x=0, and last x=num_ba-1)
    num_x = param2val1['num_partitions'] * param2val1['num_iterations']
    xticks = np.arange(0, num_x + x_step, x_step)
    yticks = np.linspace(0.5, 1.00, 6, endpoint=True).round(2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim([0.5, y_max])
    ax.set_xlim([0, xticks[-1]])
    if is_grid:
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)

    # plot
    x = np.arange(num_x)
    for n, (d1, d2, d3, dof) in enumerate(zip(d1s, d2s, d3s, dofs)):
        num_trajectories = len(d1)
        palette = iter(sns.color_palette('hls', num_trajectories))
        for num_cats in plot_num_cats_list:
            c = next(palette)
            # ba mean
            ba_means = np.asarray(d1[num_cats])
            ax.plot(x, ba_means, color=c,
                    label='num_cats={}'.format(num_cats) if n == 0 else '_nolegend_',
                    linestyle='-' if n == solid_line_id else ':')
            # ba confidence interval
            ba_sems = np.asarray(d2[num_cats])
            q = (1 - confidence) / 2.
            margins = ba_sems * stats.t.ppf(q, dof)
            ax.fill_between(x, ba_means - margins, ba_means + margins, color=c, alpha=0.2)
            # max_ba
            if d3 is not None and plot_max_ba:
                ax.axhline(y=d3[num_cats], linestyle='dashed', color=c, alpha=0.5)
    if legend:
        plt.legend(loc='upper left', frameon=False, fontsize=fontsize)

    plt.show()


def make_comparison_title(param2val1, param2val2, solid_line_id):
    solid1 = 'solid' if solid_line_id == 0 else 'dashed'
    solid2 = 'solid' if solid_line_id == 1 else 'dashed'
    #
    res = 'Comparison between\n'
    for param, val1 in param2val1.items():
        if param == 'param_name':
            continue
        val2 = param2val2[param]
        if val1 != val2:
            res += '{}={} ({}) vs. {} ({})\n'.format(param, val1, solid1, val2, solid2)
    return res
