import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from treetransitions.toy_data import ToyData
from treetransitions.params import DefaultParams, ObjectView
from treetransitions.utils import calc_kl_divergence

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (10, 10)
TITLE_FONTSIZE = 10

NUM_CATS = 32

DefaultParams.num_seqs = [2 * 10 ** 6]
DefaultParams.num_cats_list = [[2, 4, 8, 16, 32, 64, 128, 256, 512]]
DefaultParams.truncate_num_cats = [NUM_CATS]
DefaultParams.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


def plot_kld_trajs(ys, title, xticklabels):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('num_cats')
    ax.set_ylabel('KL-divergence')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0.0, 0.2])
    num_xticklabels = len(xticklabels)
    ax.set_xticks(np.arange(num_xticklabels))
    ax.set_xticklabels(xticklabels)
    # plot
    ax.plot(ys)
    #
    plt.tight_layout()
    plt.show()


for param2vals in list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params, max_ba=False)

    # mean_kl-divergences
    w2freq = Counter(toy_data.word_sequences_mat[:, -1])
    mean_klds = []
    for num_cats in params.num_cats_list:
        cat2legals = toy_data.num_cats2cat2legals[num_cats]
        cat_klds = []
        for cat, legals in cat2legals.items():
            c = Counter(legals)
            p = np.asarray(list(c.values()))
            q = np.asarray([w2freq[w] for w in c.keys()])
            kld = calc_kl_divergence(p / p.sum(), q / q.sum())
            cat_klds.append(kld)
        #
        mean_kld = np.mean(cat_klds)
        mean_klds.append(mean_kld)
        print('num_cats={:>2} kld={:.4f}'.format(num_cats, np.mean(cat_klds)))

    plot_kld_trajs(mean_klds,
                   title='truncate_list={}'.format(params.truncate_list),
                   xticklabels=params.num_cats_list)
    print('------------------------------------------------------')