import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from treetransitions.toy_data import ToyData
from treetransitions.params import Params, ObjectView
from treetransitions.utils import calc_ba

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (8, 5)

NUMS_SPLITS = 8

NUM_CATS = 32

Params.num_seqs = [1 * 10 ** 6]
Params.num_cats_list = [[NUM_CATS]]
Params.legal_probs = [[0.5, 0.5], [1.0, 1.0]]
Params.num_non_probes_list = [[1024]]
Params.num_contexts = [128]


def calc_ba_from_sequences_chunk(seqs_chunk, d):  # BOW model with window_size=1 is equivalent to term-by-window model
    for seq in seqs_chunk:
        assert len(seq) == 2
        xw, yw = seq
        vocab_id = toy_data.word2id[yw]
        try:
            d[xw][vocab_id] += 1
        except KeyError:   # x-word is not a probe
            continue
    # ba
    p_acts = [d[p] for p in probes]
    ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
    return ba


def plot_ba_trajs(tr2y, x, title, fontsize=16):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=fontsize)
    ax.set_xlabel('Number of Sequences in Partition', fontsize=fontsize)
    ax.set_ylabel('Balanced Accuracy', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.yaxis.grid(True)
    # ax.set_xticks([x[0], x[-1]])
    # ax.set_xticklabels([x[0], x[-1]])
    ax.set_ylim([0.5, 0.7])
    # plot
    for tr, y in tr2y.items():
        ax.plot(x, y, label=r'$P_1$={}'.format(tr))
    #
    plt.legend(frameon=False, loc='lower right', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


sp2bas = {tuple(sp): [0.5] for sp in Params.legal_probs}
sp2num_windows = {tuple(sp): [0] for sp in Params.legal_probs}
for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    sp = tuple(params.legal_probs)

    # toy data
    toy_data = ToyData(params, max_ba=False)
    probes = toy_data.x_words
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cats = set(probe2cat.values())

    probe2act = {p: np.zeros(toy_data.num_vocab) for p in probes}
    xi = 0
    for rows in np.vsplit(np.asarray(toy_data.word_sequences_mat), NUMS_SPLITS):
        ba = calc_ba_from_sequences_chunk(rows, probe2act)
        xi += len(rows)
        sp2bas[sp].append(ba)
        sp2num_windows[sp].append(xi)
        print('xi={} ba={:.2f}'.format(xi, ba))
    print('------------------------------------------------------')


# plot
plot_ba_trajs(sp2bas, sp2num_windows[sp],
              title='Semantic category information in artificial input\n'
                    'captured by term-by-window co-occurrence matrix'
                    '\n with size=1')