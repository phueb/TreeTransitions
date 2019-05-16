import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from treetransitions.toy_data import ToyData
from treetransitions.params import Params, ObjectView
from treetransitions.utils import calc_ba

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (5, 5)
TITLE_FONTSIZE = 10

NUMS_SPLITS = 4

NUM_CATS = 32
TRUNCATE_TYPE = 'legals'
TRUNCATE_CONTROL = True

Params.num_seqs = [5 * 10 ** 6]
Params.num_cats_list = [[NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[0.5, 1.0], [1.0, 1.0]]
Params.truncate_type = [TRUNCATE_TYPE]
Params.truncate_control = [TRUNCATE_CONTROL]


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


def plot_ba_trajs(tr2y, x, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Location Toy Data')
    ax.set_ylabel('Balanced Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_xticks([x[0], x[-1]])
    ax.set_xticklabels([x[0], x[-1]])
    ax.set_ylim([0, 1.0])
    # plot
    for tr, y in tr2y.items():
        ax.plot(x, y, label='truncate={}'.format(tr))
    #
    plt.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    plt.show()


truncate2bas = {tuple(tr): [0.5] for tr in Params.truncate_list}
truncate2num_windows = {tuple(tr): [0] for tr in Params.truncate_list}
for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    truncate = tuple(params.truncate_list)

    # toy data
    toy_data = ToyData(params, max_ba=False)
    probes = toy_data.probes
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cats = set(probe2cat.values())

    probe2act = {p: np.zeros(toy_data.num_vocab) for p in probes}
    xi = 0
    for rows in np.vsplit(np.asarray(toy_data.word_sequences_mat), NUMS_SPLITS):
        ba = calc_ba_from_sequences_chunk(rows, probe2act)
        xi += len(rows)
        truncate2bas[truncate].append(ba)
        truncate2num_windows[truncate].append(xi)
        print('xi={} ba={:.2f}'.format(xi, ba))
    print('------------------------------------------------------')


# plot
plot_ba_trajs(truncate2bas, truncate2num_windows[truncate],
              title='model=bag-of-words\n'
                    'truncate_type={} truncate_control={}'.format(
                  TRUNCATE_TYPE, TRUNCATE_CONTROL))