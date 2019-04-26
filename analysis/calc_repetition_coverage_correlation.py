import numpy as np
import matplotlib.pyplot as plt


from treetransitions.toy_data import ToyData
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (6, 6)
TITLE_FONTSIZE = 12

NUM_CATS = 32

DefaultParams.num_tokens = [2 * 10 ** 6]
DefaultParams.num_cats_list = [[NUM_CATS]]
DefaultParams.truncate_num_cats = [NUM_CATS]
DefaultParams.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


def plot_corrs(ys, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Category')
    ax.set_ylabel('Correlation betweeen coverage & repetition')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0.8, 1.0])
    ax.set_xticks([])
    ax.set_xticklabels([])
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
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]

    # init
    print('Counting...')
    cats = set(probe2cat.values())
    word2prev_cat2count = {w: {cat: 0 for cat in cats} for w in toy_data.vocab}
    word2prev_word2count = {w: {w: 0 for w in toy_data.vocab} for w in toy_data.vocab}
    for n, token in enumerate(toy_data.tokens[1:]):
        prev_token = toy_data.tokens[n - 1]
        try:
            prev_cat = probe2cat[prev_token]
        except KeyError:
            prev_cat = None
        #
        word2prev_cat2count[token][prev_cat] += 1
        word2prev_word2count[token][prev_token] += 1

    corrs = []
    for cat in cats:
        cat_probes = [p for p in toy_data.probes if probe2cat[p] == cat]
        cat_repetitions = []
        cat_coverages = []
        for word in toy_data.vocab:
            repetition = word2prev_cat2count[word][cat]  # how many times word occurs after category member
            coverage = np.count_nonzero([word2prev_word2count[word][w] for w in cat_probes])
            if repetition > 0.0:
                cat_repetitions.append(repetition)
                cat_coverages.append(coverage)

        corr = np.corrcoef(cat_repetitions, cat_coverages)[0, 1]
        print(corr)
        corrs.append(corr)

    plot_corrs(corrs,
               title='truncate_list={}'.format(params.truncate_list))
    print('------------------------------------------------------')