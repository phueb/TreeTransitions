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
COLORS = ['red', 'blue']

NUM_CATS = 32

DefaultParams.num_seqs = [1 * 10 ** 6]
DefaultParams.num_cats_list = [[NUM_CATS]]
DefaultParams.truncate_num_cats = [NUM_CATS]
DefaultParams.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


def make_bigram_count_mat(id_seqs_mat, num_vocab):
    assert id_seqs_mat.shape[1] == 2  # works with bi-grams only
    print('Making In-Out matrix with shape={}...'.format(id_seqs_mat.shape))
    res = np.zeros((num_vocab, num_vocab))
    for row_id, col_id in zip(id_seqs_mat[:, 0], id_seqs_mat[:, 1]):
        res[row_id, col_id] += 1
    return res


# fig1
fig1, ax1 = plt.subplots(figsize=FIGSIZE, dpi=None)
plt.title('Context Word Statistics\n(1 line per category)', fontsize=TITLE_FONTSIZE)
ax1.set_xlabel('Category Member')
ax1.set_ylabel('Mean of context word counts')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(axis='both', which='both', top=False, right=False)

# fig2
fig2, ax2 = plt.subplots(figsize=FIGSIZE, dpi=None)
plt.title('Context Word Statistics\n(1 line per category)', fontsize=TITLE_FONTSIZE)
ax2.set_xlabel('Category Member')
ax2.set_ylabel('Variance of context word counts')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='both', which='both', top=False, right=False)

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

    bigram_count_mat = make_bigram_count_mat(toy_data.id_sequences_mat, toy_data.num_vocab)

    # plot statistic not collapsing over categories
    label = 'truncate_list={}'.format(params.truncate_list)
    for cat in cats:
        cat_probes = [p for p in toy_data.probes if probe2cat[p] == cat]
        y1 = np.sort([bigram_count_mat[toy_data.word2id[p]].mean() for p in cat_probes])
        y2 = np.sort([bigram_count_mat[toy_data.word2id[p]].var() for p in cat_probes])
        # plot
        ax1.plot(y1,
                 alpha=1.0,
                 label=label if cat == 0 else '_nolegend_',
                 color=COLORS[DefaultParams.truncate_list.index(params.truncate_list)])
        ax2.plot(y2,
                 alpha=1.0,
                 label=label if cat == 0 else '_nolegend_',
                 color=COLORS[DefaultParams.truncate_list.index(params.truncate_list)])

    print('------------------------------------------------------')


ax1.legend(frameon=False)
ax2.legend(frameon=False)
plt.tight_layout()
plt.show()