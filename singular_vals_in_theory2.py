import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


CENTER_AND_SCALE = False  # singular vals are only different if matrix is not centered + scaled


def plot_data(co_occurrence_mats, fontsize=14):
    # fig
    fig, axarr = plt.subplots(2, 2, figsize=(10, 10), dpi=None)
    # axarr row 1: computing singular value 1
    for ax, xys, p in zip(axarr[0], co_occurrence_mats, ['p1', 'p2']):
        ax.set_title(p, fontsize=fontsize)
        ax.set_xlabel('window 1 co-occurrence\nfrequency', fontsize=fontsize)
        ax.set_ylabel('window 2 co-occurrence\nfrequency', fontsize=fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # ticks
        xticks = np.arange(-1, 12)
        yticks = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels([val if val != 0 else '' for val in xticks])
        ax.set_yticks(yticks)
        ax.set_yticklabels([val if val != 0 else '' for val in yticks])
        ax.set_xlim([np.min(xticks), np.max(xticks)])
        ax.set_ylim([np.min(xticks), np.max(xticks)])
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        # plot
        ax.scatter(x=xys[:, 0], y=xys[:, 1], zorder=3, color='black')
        ax.plot([0, 10], [0, 10], color='grey', linestyle=':')
        # projections
        ax.plot(np.array([0, 5]) + 0.0, np.array([0, 5]) - 0.0, color='red', linestyle='-')
        ax.plot(np.array([0, 5]) + 0.1, np.array([0, 5]) - 0.1, color='red', linestyle='-')
        ax.plot(np.array([0, 5]) + 0.2, np.array([0, 5]) - 0.2, color='red', linestyle='-')
        ax.plot(np.array([0, 5]) + 0.3, np.array([0, 5]) - 0.3, color='red', linestyle='-')

    # axarr row 2: computing singular value 2
    for ax, xys, p in zip(axarr[1], co_occurrence_mats, ['p1', 'p2']):
        ax.set_title(p, fontsize=fontsize)
        ax.set_xlabel('window 1 co-occurrence\nfrequency', fontsize=fontsize)
        ax.set_ylabel('window 2 co-occurrence\nfrequency', fontsize=fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # ticks
        xticks = np.arange(-1, 12)
        yticks = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels([val if val != 0 else '' for val in xticks])
        ax.set_yticks(yticks)
        ax.set_yticklabels([val if val != 0 else '' for val in yticks])
        ax.set_xlim([np.min(xticks), np.max(xticks)])
        ax.set_ylim([np.min(xticks), np.max(xticks)])
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        # plot
        ax.scatter(x=xys[:, 0], y=xys[:, 1], zorder=3, color='black')
        ax.plot([0, 10], [10, 0], color='grey', linestyle=':')
        # projections
        if 10 in xys:
            ax.plot(np.array([0, 5]) + 0.0, np.array([10, 5]) + 0.0, color='red', linestyle='-')
            ax.plot(np.array([1, 5]) + 0.1, np.array([9,  5]) + 0.1, color='red', linestyle='-')

            ax.plot(np.array([5, 10]) - 0.1, np.array([5, 0]) - 0.1, color='red', linestyle='-')
            ax.plot(np.array([5,  9]) - 0.2, np.array([5, 1]) - 0.2, color='red', linestyle='-')
        else:
            ax.plot(np.array([3, 5]) + 0.0, np.array([7, 5]) + 0.0, color='red', linestyle='-')
            ax.plot(np.array([4, 5]) + 0.1, np.array([6, 5]) + 0.1, color='red', linestyle='-')

            ax.plot(np.array([5, 6]) - 0.1, np.array([5, 4]) - 0.1, color='red', linestyle='-')
            ax.plot(np.array([5, 7]) - 0.2, np.array([5, 3]) - 0.2, color='red', linestyle='-')
    plt.show()


mat1 = np.array([[10, 0],
                 [9,  1],
                 [0, 10],
                 [1,  9]])

mat2 = np.array([[7, 3],
                 [6, 4],
                 [3, 7],
                 [4, 6]])

# plot
plot_data([mat1, mat2])


for term_window_co_occurrence_mat, title in [(mat1, 'p1'),
                                             (mat2, 'p2')]:

    # center
    if CENTER_AND_SCALE:
        mat = scale(term_window_co_occurrence_mat, axis=0, with_std=True, with_mean=True)
        print(mat)

    else:
        mat = term_window_co_occurrence_mat

    print(mat.sum())
    print(mat.var())
    print([col.mean() for col in mat.T])
    print([col.var() for col in mat.T])

    # SVD on term_window_co_occurrence_mat
    u, s, v = np.linalg.svd(mat, compute_uv=True, full_matrices=False)
    print('svls', ' '.join(['{:>6.2f}'.format(si) for si in s]))
    print('sum of svls={:,}'.format(np.sum(s)))
    print(u)
    print(v)
    print()

    # compute singular value 1
    v1 = np.array([5, 5]) / np.linalg.norm([5, 5])  # must be unit-length
    theoretical_s1 = np.linalg.norm(np.dot(term_window_co_occurrence_mat, v1), 2)
    print('theoretical s1={}'.format(theoretical_s1))

    # compute singular value 2
    v2 = np.array([5, -5]) / np.linalg.norm([5, -5])  # must be unit-length and orthogonal to v1
    theoretical_s2 = np.linalg.norm(np.dot(term_window_co_occurrence_mat, v2), 2)
    print('theoretical s2={}'.format(theoretical_s2))
