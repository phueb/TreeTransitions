import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData
from treetransitions.utils import make_term_by_window_mat

from ludwigcluster.utils import list_all_param2vals


# fig
PLOT_NUM_SVS = 64
FONTSIZE = 16
FIGSIZE = (5, 5)
DPI = None


Params.mutation_prob = [0.01]
Params.num_seqs = [1 * 10 ** 6]  # TODO
Params.num_partitions = [2]
Params.legal_probs = [[0.5, 1.0]]
Params.num_non_probes_list = [[1024]]
Params.num_contexts = [128]  # TODO if this is high, then singular values are no different - why?


def plot_comparison(ys, params):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('number of syntactic categories={}\nnumber of context words={}'.format(
        len(params.num_non_probes_list), params.num_contexts))
    ax.set_ylabel('Singular Value', fontsize=FONTSIZE)
    ax.set_xlabel('Singular Dimension', fontsize=FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    x = np.arange(PLOT_NUM_SVS)
    # plot
    labels = iter([r'$P_1={}$'.format(sp) for sp in params.legal_probs])
    for n, y in enumerate(ys):
        ax.plot(x, y, label=next(labels) or 'partition {}'.format(n + 1), linewidth=2)
    ax.legend(loc='upper right', frameon=False, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # toy_data
    params = ObjectView(param2val)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)
    toy_data = ToyData(params, max_ba=False, make_sequences=True)

    # singular_vals
    singular_vals = []
    for word_seq_mat_chunk in np.vsplit(toy_data.word_sequences_mat, params.num_partitions):
        term_by_window_mat, _, _ = make_term_by_window_mat(word_seq_mat_chunk, toy_data)
        normalized = normalize(term_by_window_mat, axis=1, norm='l2', copy=False)
        u, s, v = np.linalg.svd(normalized, full_matrices=False)
        singular_vals.append(s[:PLOT_NUM_SVS])

    # plot
    plot_comparison(singular_vals, params)