import numpy as np

import matplotlib.pyplot as plt

"""
compute singular values for term-by-window co-occurrence matrices defined by the legals matrices belows.
"""

NUM_SEQUENCES = 100000  # each partition has exactly the same number of sequences (this must be true)


def plot_comparison(ys, fontsize=12, figsize=(5, 5)):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=None)
    plt.title('SVD of hypothetical\nterm-by-window co-occurrence matrix', fontsize=fontsize)
    ax.set_ylabel('Singular value', fontsize=fontsize)
    ax.set_xlabel('Singular Dimension', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    x = np.arange(12) + 1  # num columns
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    # plot
    for n, y in enumerate(ys):
        ax.plot(x, y, label='partition {}'.format(n + 1), linewidth=2)
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


part1_legals_mat = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]])


part2_legals_mat = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

ys = []
for mat in [part1_legals_mat, part2_legals_mat, ]:
    # make term_window_co_occurrence_mat
    term_window_co_occurrence_mat = np.zeros_like(mat)
    nonzero_ids = np.nonzero(mat)
    num_nonzeros = np.sum(mat)
    for _ in range(NUM_SEQUENCES):
        idx = np.random.choice(num_nonzeros, size=1)
        i = nonzero_ids[0][idx].item()
        j = nonzero_ids[1][idx].item()
        term_window_co_occurrence_mat[i, j] += 1
    print(term_window_co_occurrence_mat)
    print('sum={:,}'.format(term_window_co_occurrence_mat.sum()))  # sums must match
    print('var={:,}'.format(term_window_co_occurrence_mat.var()))
    # SVD on term_window_co_occurrence_mat
    s = np.linalg.svd(term_window_co_occurrence_mat, compute_uv=False)
    print('svls', ' '.join(['{:>6.2f}'.format(si) for si in s]))
    print('sum of svls={:,}'.format(np.sum(s)))
    print()
    # collect
    ys.append(s)

plot_comparison(ys)

