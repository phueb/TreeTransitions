import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData
from treetransitions.utils import make_term_by_window_mat

from ludwigcluster.utils import list_all_param2vals


# fig
NUM_SVS = 64
FONTSIZE = 16
FIGSIZE = (5, 5)
DPI = None


Params.mutation_prob = [0.01]
Params.num_seqs = [1 * 10 ** 6]
Params.num_partitions = [2]
Params.legal_probs = [[0.5, 1.0]]
Params.num_non_probes_list = [[1024, 1024, 1024]]
Params.num_contexts = [1024]


def plot_distributional_sim_trajs(params, *ys):
    # fig
    x = [0, 1]
    fig, ax = plt.subplots(dpi=None, figsize=(8, 8))
    plt.title('Distributional similarity of syntactic categories'
              '\nnumber of SVD modes={}'.format(NUM_SVS), fontsize=FONTSIZE)
    ax.set_ylabel('Cosine Similarity', fontsize=FONTSIZE)
    ax.set_xlabel('Partition', fontsize=FONTSIZE)
    # ax.set_ylim([0, np.max(pos_member_sims) + 0.01])
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    for y in ys:
        ax.plot(x, y)
    fig.tight_layout()
    plt.show()
    plt.show()


for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):
    params = ObjectView(param2val)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)
    toy_data = ToyData(params, max_ba=False, make_sequences=True)

    # compute sims between representations
    y0 = []
    y1 = []
    y2 = []
    for word_seq_mat_chunk in np.vsplit(toy_data.word_sequences_mat, params.num_partitions):
        term_by_window_mat, all_xws, all_yws = make_term_by_window_mat(word_seq_mat_chunk, toy_data)
        normalized = normalize(term_by_window_mat, axis=1, norm='l2', copy=False)
        u, s, v = np.linalg.svd(normalized, full_matrices=False, k=NUM_SVS, return_singular_vectors='u')
        # get reps
        bool_ids0 = [True if yw in toy_data.name2words[0][1] else False for yw in all_yws]
        bool_ids1 = [True if yw in toy_data.name2words[1][1] else False for yw in all_yws]
        bool_ids2 = [True if yw in toy_data.name2words[2][1] else False for yw in all_yws]
        syn_cat_members0 = u[bool_ids0]
        syn_cat_members1 = u[bool_ids1]
        syn_cat_members2 = u[bool_ids2]
        # sim
        sim0 = np.corrcoef(syn_cat_members0, rowvar=True).mean().item()
        sim1 = np.corrcoef(syn_cat_members1, rowvar=True).mean().item()
        sim2 = np.corrcoef(syn_cat_members2, rowvar=True).mean().item()
        y0.append(sim0)
        y1.append(sim1)
        y2.append(sim2)
        # plot
        plot_distributional_sim_trajs(params, y0, y1, y2)