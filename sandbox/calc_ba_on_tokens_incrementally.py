import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from treetransitions.figs import plot_ba_trajectories_simple
from treetransitions.toy_data import ToyData
from treetransitions.params import param2default, param2requests
from treetransitions.utils import calc_ba
from treetransitions.job import Params

from ludwig.results import gen_all_param2vals


NUMS_SPLITS = 8
NUM_CATS = 32


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


sp2bas = {tuple(sp): [0.5] for sp in param2requests['legal_probabilities']}
sp2num_windows = {tuple(sp): [0] for sp in param2requests['legal_probabilities']}
for param2val in gen_all_param2vals(param2requests, param2default):

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    sp = tuple(params.legal_probabilities)

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
plot_ba_trajectories_simple(sp2bas, sp2num_windows[sp],
                            title='Semantic category information in artificial input\n'
                                  'captured by term-by-window co-occurrence matrix'
                                  '\n with size=1')