import numpy as np
import matplotlib.pyplot as plt


from treetransitions.toy_data import ToyData
from treetransitions.params import Params, ObjectView

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (6, 6)
TITLE_FONTSIZE = 12

NUM_CATS = 32

Params.num_seqs = [2 * 10 ** 6]
Params.num_cats_list = [[NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


def plot_corrs(ys, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Category')
    ax.set_ylabel('Correlation between coverage & repetition')
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


def make_bigram_count_mat(id_seqs_mat, num_vocab):
    assert id_seqs_mat.shape[1] == 2  # works with bi-grams only
    print('Making In-Out matrix with shape={}...'.format(id_seqs_mat.shape))
    res = np.zeros((num_vocab, num_vocab))
    for row_id, col_id in zip(id_seqs_mat[:, 0], id_seqs_mat[:, 1]):
        # freq
        res[row_id, col_id] += 1
    return res


for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params, max_ba=False)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cats = set(probe2cat.values())

    bigram_count_mat = make_bigram_count_mat(toy_data.id_sequences_mat, toy_data.num_vocab)

    corrs = []
    for cat in cats:
        cat_probes = [p for p in toy_data.probes if probe2cat[p] == cat]
        cat_repetitions = []
        cat_coverages = []
        for col in bigram_count_mat.T:  # iterate over contexts
            num_after_cat = [count if toy_data.vocab[word_id] in cat_probes else 0
                             for word_id, count in enumerate(col)]
            repetition = np.sum(num_after_cat)  # how often context occurs after category
            coverage = np.count_nonzero(num_after_cat)  # how many members context occurs after
            #
            cat_repetitions.append(repetition)
            cat_coverages.append(coverage)

        corr = np.corrcoef(cat_repetitions, cat_coverages)[0, 1]
        print(corr)
        corrs.append(corr)

    plot_corrs(corrs,
               title='truncate_list={}'.format(params.truncate_list))
    print('------------------------------------------------------')