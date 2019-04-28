import numpy as np
import matplotlib.pyplot as plt


from treetransitions.toy_data import ToyData
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


"""
note: this analysis is only informative for toy data where tokens are sampled uniformly from legals.
this means, this analysis shouldn't be used on CHILDES where this is not guaranteed

"""


FIGSIZE = (6, 6)
TITLE_FONTSIZE = 12

NUM_CATS = 32

DefaultParams.num_seqs = [2 * 10 ** 6]
DefaultParams.num_cats_list = [[NUM_CATS]]
DefaultParams.truncate_num_cats = [NUM_CATS]
DefaultParams.truncate_list = [[0.1, 0.1], [1.0, 1.0]]


def plot_rep_cov_density(mats, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title('Coverage-Repetition Density', fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Category Member')
    ax.set_ylabel('Probability')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([0, 2 / NUM_CATS])
    # plot
    ax.axhline(y=1 / NUM_CATS, color='grey')
    ax.axvline(x=NUM_CATS / 2, color='grey')
    for mat, lab in zip(mats, labels):
        row_sum = mat.sum(axis=0)
        ax.plot(row_sum / row_sum.sum(), label=lab)  # TODO test
    #
    plt.legend(frameon=False)
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


cov_rep_mats = []
labels = []
for param2vals in list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params, max_ba=False)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cats = set(probe2cat.values())
    num_cats = len(cats)
    num_members = params.parent_count // NUM_CATS

    bigram_count_mat = make_bigram_count_mat(toy_data.id_sequences_mat, toy_data.num_vocab)

    # cov_rep_mat
    print('Making cov_rep_mat...')
    cov_rep_mat = np.zeros((num_cats, num_members))
    for cat in cats:
        cat_probes = [p for p in toy_data.probes if probe2cat[p] == cat]
        row = [bigram_count_mat[toy_data.word2id[p]].sum() for p in cat_probes]
        cov_rep_mat[cat] = np.sort(row)

    # collect
    label = 'truncate_list={}'.format(params.truncate_list)
    labels.append(label)
    cov_rep_mats.append(cov_rep_mat)
    print('------------------------------------------------------')


# plot
plot_rep_cov_density(cov_rep_mats, labels)