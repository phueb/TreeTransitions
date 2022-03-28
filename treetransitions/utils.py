from functools import partial

import numpy as np
from bayes_opt import BayesianOptimization
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from results.plot_ba_trajectories import TOLERANCE


def calc_ba(probe_sims, probes, probe2cat, num_opt_init_steps=1, num_opt_steps=10):
    def calc_signals(_probe_sims, _labels, thr):  # vectorized algorithm is 20X faster
        probe_sims_clipped = np.clip(_probe_sims, 0, 1)
        probe_sims_clipped_triu = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
        predictions = np.zeros_like(probe_sims_clipped_triu, int)
        predictions[np.where(probe_sims_clipped_triu > thr)] = 1
        #
        tp = float(len(np.where((predictions == _labels) & (_labels == 1))[0]))
        tn = float(len(np.where((predictions == _labels) & (_labels == 0))[0]))
        fp = float(len(np.where((predictions != _labels) & (_labels == 0))[0]))
        fn = float(len(np.where((predictions != _labels) & (_labels == 1))[0]))
        return tp, tn, fp, fn

    # gold_mat
    if not len(probes) == probe_sims.shape[0] == probe_sims.shape[1]:
        raise RuntimeError(len(probes), probe_sims.shape[0], probe_sims.shape[1])
    num_rows = len(probes)
    num_cols = len(probes)
    gold_mat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        probe1 = probes[i]
        for j in range(num_cols):
            probe2 = probes[j]
            if probe2cat[probe1] == probe2cat[probe2]:
                gold_mat[i, j] = 1

    # define calc_signals_partial
    labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
    calc_signals_partial = partial(calc_signals, probe_sims, labels)

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    sims_mean = np.mean(probe_sims).item()
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
    bo = BayesianOptimization(calc_probes_ba, {'thr': (0.0, 1.0)}, verbose=False)
    bo.explore(
        {'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=num_opt_init_steps, n_iter=num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = calc_probes_ba(best_thr)
    res = np.mean(results)
    return res


def cluster(m, original_row_words=None, original_col_words=None):
    print('Clustering...')
    #
    lnk0 = linkage(pdist(m))
    dg0 = dendrogram(lnk0,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)
    res = m[dg0['leaves'], :]  # reorder rows
    #
    lnk1 = linkage(pdist(m.T))
    dg1 = dendrogram(lnk1,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)

    res = res[:, dg1['leaves']]  # reorder cols
    #
    if original_row_words is None and original_col_words is None:
        return res
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return res, row_labels, col_labels


def correct_artifacts(df):
    """
    remove negative-going dips in balanced accuracy curves
    """
    for bas in df.values.T:
        num_bas = len(bas)
        for i in range(num_bas - 2):
            val1, val2, val3 = bas[[i, i+1, i+2]]
            if (val1 - TOLERANCE) > val2 < (val3 - TOLERANCE):
                bas[i+1] = np.mean([val1, val3])
    return df