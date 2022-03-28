import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
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


def plot_ba_trajectories(summary: Tuple[dict, dict, int, int],
                         label: str,
                         y_max: float = 1.0,
                         is_grid: bool = False,
                         legend: bool = True,
                         title: Optional[str] = None,
                         fontsize: int = 16,
                         fig_size: Tuple[int, int] = (8, 6),
                         ):

    # get data
    num_cats2avg_ba, num_cats2std_ba, eval_steps, num_results = summary

    # title
    if title is None:
        title = label

    # fig
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)
    plt.title(title, fontsize=fontsize)
    ax.set_xlabel('Training Step',
                  fontsize=fontsize)
    ax.set_ylabel(f'Balanced Accuracy +/- Std Dev.',
                  fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    yticks = np.linspace(0.5, 1.00, 6, endpoint=True).round(2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim([0.5, y_max])
    if is_grid:
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)

    # plot
    num_trajectories = len(num_cats2avg_ba)
    palette = iter(sns.color_palette('hls', num_trajectories))
    for num_cats in sorted(num_cats2avg_ba):
        c = next(palette)
        # ba mean
        ba_means = num_cats2avg_ba[num_cats]
        ax.plot(eval_steps,
                ba_means,
                color=c,
                label=f'number of categories ={num_cats}',
                )
        # ba std
        std_half = num_cats2std_ba[num_cats] / 2
        ax.fill_between(eval_steps,
                        ba_means - std_half,
                        ba_means + std_half,
                        color=c,
                        alpha=0.2,
                        )

    if legend:
        plt.legend(loc='lower right',
                   frameon=False,
                   fontsize=fontsize)

    plt.show()
