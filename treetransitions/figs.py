import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from typing import Optional, Tuple, List


def plot_heatmap(mat: np.array,
                 y_tick_labels: List[str],
                 x_tick_labels: List[str],
                 title: str = '',
                 x_label: Optional[str] = None,
                 y_label: Optional[str] = None,
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
              interpolation='nearest',
              vmax=1.0,
              vmin=0.0,
              )
    ax.set_xlabel(x_label or 'Y-Words', fontsize=fontsize)
    ax.set_ylabel(y_label or 'X-Words', fontsize=fontsize)

    # ticks
    num_rows, num_cols = mat.shape
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(x_tick_labels, rotation=90, fontsize=tick_label_font_size)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(y_tick_labels,  # no need to reverse (because no extent is set)
                            rotation=0, fontsize=tick_label_font_size)

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    plt.show()


def plot_ba_trajectories(summary: Tuple[dict, dict, dict, int],
                         plot_num_cats_list: Tuple[int],
                         y_max: float = 1.0,
                         x_step: int = 5,
                         is_grid: bool = False,
                         confidence: float = 0.95,
                         solid_line_id: int = 0,  # 0 or 1 to flip which line is solid
                         legend: bool = True,
                         title: Optional[str] = None,
                         fontsize: int = 16,
                         fig_size: Tuple[int, int] = (8, 8),
                         ):

    # get data
    num_cats2ba_means1, num_cats2ba_sems1, param2val1, num_results1 = summary
    d1s = [num_cats2ba_means1]
    d2s = [num_cats2ba_sems1]
    dofs = [num_results1 - 1]

    # title
    if title is None:
        title_base = ''
        for param, val in sorted(param2val1.items(), key=lambda i: i[0]):
            title_base += '{}={}\n'.format(param, val)
        title = title_base + '\nn={}'.format(num_results1)

    # fig
    fig, ax = plt.subplots(figsize=fig_size, dpi=None)
    plt.title(title, fontsize=fontsize)
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Balanced Accuracy +/-\n{}%-Confidence Interval'.format(confidence * 100), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ticks (ba is calculated before training update, so first ba is x=0, and last x=num_ba-1)
    num_x = param2val1['num_parts'] * param2val1['num_iterations']
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
    for n, (d1, d2, dof) in enumerate(zip(d1s, d2s, dofs)):
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
