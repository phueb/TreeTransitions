import multiprocessing as mp


from treetransitions.toy_data import ToyData
from treetransitions.params import Params, ObjectView

from ludwigcluster.utils import list_all_param2vals


FIGSIZE = (6, 6)
TITLE_FONTSIZE = 12

NUM_CATS = 32

Params.num_seqs = [1 * 10 ** 4]
Params.num_cats_list = [[NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


def calc_overlap(d, cat):
    """
    quantify overlap between contexts of one category with contexts of all other categories
    """
    other_cats = [c for c in cats if c != cat]
    num = 0
    num_total = 0
    for context_word1 in d[cat]:
        for other_cat in other_cats:
            for context_word2 in d[other_cat]:
                if context_word1 == context_word2:
                    num += 1
                num_total += 1

    res = num / num_total  # division controls for greater number of probes in partition 1
    print(res)
    return res


for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params, max_ba=False)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cats = set(probe2cat.values())

    # cat2contexts
    cat2contexts = {cat: [] for cat in cats}
    for pw, cw in toy_data.word_sequences_mat:
        cat = probe2cat[pw]
        cat2contexts[cat].append(cw)
    #
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(calc_overlap, args=(cat2contexts, cat)) for cat in cats]
    overlaps = []
    print('Calculating...')
    try:
        for r in results:
            overlap = r.get()
            overlaps.append(overlap)
        pool.close()
    except KeyboardInterrupt:
        pool.close()
        raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
    print('mean overlap={:,}'.format(sum(overlaps) / len(overlaps)))
    print()

    print('------------------------------------------------------')