import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from treetransitions.toy_data import ToyData
from treetransitions.params import Params, ObjectView
from treetransitions.utils import calc_ba

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (5, 5)
TITLE_FONTSIZE = 10

NUMS_SPLITS = 8

NUM_CATS = 32

Params.num_seqs = [1 * 10 ** 5]
Params.num_cats_list = [[NUM_CATS]]
Params.min_num_cats = [NUM_CATS]
Params.reverse = [False]
Params.mutation_prob = [0.05]
Params.template_noise = [0.6]


def calc_ba_from_sequences_chunk(seqs_chunk, d):
    for seq in seqs_chunk:
        assert len(seq) == 2
        p, c = seq
        vocab_id = toy_data.word2id[c]
        try:
            d[p][vocab_id] += 1
        except KeyError:  # probe word was replaced by y_word (due to truncate_type='probes')
            continue
    # ba
    p_acts = [d[p] for p in probes]
    ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
    return ba


def plot_ba_trajs(r2y, x, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Location Toy Data')
    ax.set_ylabel('Balanced Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_xticks([x[0], x[-1]])  # TODO test
    ax.set_xticklabels([x[0], x[-1]])
    # plot
    for reverse, y in r2y.items():
        ax.plot(x, y, label='reverse={}'.format(reverse))
    #
    plt.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    plt.show()


# params
param2vals = list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'})[0]
params = ObjectView(param2vals)
for k, v in sorted(params.__dict__.items()):
    print(k, v)

# toy data
toy_data = ToyData(params)
probes = toy_data.x_words
probe2cat = toy_data.num_cats2xw2cat[NUM_CATS]
cats = set(probe2cat.values())

reverse_list = [False, True]
reverse2bas = {reverse: [0.5] for reverse in reverse_list}
reverse2num_windows = {reverse: [0] for reverse in reverse_list}
for reverse in reverse_list:
    probe2act = {p: np.zeros(toy_data.num_vocab) for p in probes}
    xi = 0
    seq_mat = toy_data.word_sequences_mat if not reverse else toy_data.word_sequences_mat[::-1]
    for rows in np.vsplit(np.asarray(seq_mat), NUMS_SPLITS):
        ba = calc_ba_from_sequences_chunk(rows, probe2act)
        xi += len(rows)
        reverse2bas[reverse].append(ba)
        reverse2num_windows[reverse].append(xi)
        print('xi={} ba={:.2f}'.format(xi, ba))
    print('------------------------------------------------------')


# plot
plot_ba_trajs(reverse2bas, reverse2num_windows[True],
              title='model=bag-of-words\n'
                    'mutation_prob={}\ntemplate_noise={}'.format(
                  params.mutation_prob, params.template_noise))