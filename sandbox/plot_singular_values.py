import numpy as np
from sklearn.preprocessing import normalize

from ludwig.results import gen_all_param2vals

from treetransitions.figs import plot_comparison
from treetransitions.params import param2requests, param2default
from treetransitions.toy_data import ToyData
from treetransitions.utils import make_term_by_window_mat
from treetransitions.job import Params


PLOT_NUM_SVS = 8


for param2val in gen_all_param2vals(param2requests, param2default):

    # toy_data
    params = Params.from_param2val(param2val)
    toy_data = ToyData(params, max_ba=False, make_sequences=True)

    # singular values
    singular_values = []
    for word_seq_mat_chunk in np.vsplit(toy_data.word_sequences_mat, params.num_partitions):
        term_by_window_mat, _, _ = make_term_by_window_mat(word_seq_mat_chunk, toy_data)
        normalized = normalize(term_by_window_mat, axis=1, norm='l2', copy=False)
        u, s, v = np.linalg.svd(normalized, full_matrices=False)
        singular_values.append(s[:PLOT_NUM_SVS])

    # plot
    plot_comparison(singular_values, params, PLOT_NUM_SVS)