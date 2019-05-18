import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from treetransitions.params import Params, ObjectView
import matplotlib.pyplot as plt

from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (5, 5)
FONTSIZE = 8


Params.mutation_prob = [0.01]
Params.num_seqs = [1 * 10 ** 5]
Params.num_cats_list = [[2]]
Params.num_levels = [3, 4, 5, 6, 7, 8, 9, 10]
Params.truncate_num_cats = [2]
Params.learning_rate = [0.003]
Params.truncate_control = ['none']
Params.truncate_list = [[1.0, 1.0]]
Params.w = ['embeds']


def plot_sim_trajs(x, y1, y2, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=FONTSIZE)
    ax.set_xlabel('params.num_levels', fontsize=FONTSIZE)
    ax.set_ylabel('Correlation', fontsize=FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0, 1.0])
    # plot
    ax.plot(x, y1, label='between categories')
    ax.plot(x, y2, label='within categories')
    #
    plt.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.show()


x = []
y1 = []
y2 = []
for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    params.num_probes = params.num_descendants ** params.num_levels
    toy_data = ToyData(params, max_ba=False, make_tokens=False)

    legals_mat = toy_data.legals_mats[-1]

    # calc sim between legals_mat cols corresponding to different categories
    probes = toy_data.probes
    probe2cat = toy_data.num_cats2probe2cat[2]
    cats = list(set(probe2cat.values()))
    #
    acts_list = []
    for cat in cats:
        cat_probes = [p for p in probes if probe2cat[p] == cat]
        p_acts = np.asarray([legals_mat[:, toy_data.x_words.index(p)] for p in cat_probes])
        print(p_acts.shape)
        acts_list.append(p_acts)
    #
    betw_cat_mean_sim = cosine_similarity(*acts_list).mean().round(2)
    print('similarity between categories={}'.format(betw_cat_mean_sim))
    within_cat_sims = []
    for acts in acts_list:
        within_cat_sim = cosine_similarity(acts).mean().mean().round(2)
        within_cat_sims.append(within_cat_sim)
    within_cat_mean_sim = np.mean(within_cat_sims)
    print('similarity within categories={}'.format(within_cat_mean_sim))

    # collect
    x.append(params.num_levels)
    y1.append(betw_cat_mean_sim)
    y2.append(within_cat_mean_sim)
    print('------------------------------------------------------------')

assert len(Params.truncate_list) == 1  # for fig title to be accurate
assert len(Params.truncate_control) == 1  # for fig title to be accurate
plot_sim_trajs(x, y1, y2,
               title='Does increasing num_levels make num_cats=2 structure more similar?\n'
                     '(vectors representing probes are columns from legals_mat)'
                     '\ntruncate={} truncate_control={}'.format(
                   Params.truncate_list[0], Params.truncate_control[0]))