from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from treetransitions.hierarchical_data_utils import make_data, make_probe_data, calc_ba
from treetransitions import params


params.num_levels = 10
params.e = 0.2


# make tokens with hierarchical n-gram structure
vocab, tokens, ngram2legals_mat = make_data(
    params.num_tokens, params.legals_distribution, params.max_ngram_size,
    params.num_descendants, params.num_levels, params.mutation_prob, params.truncate)
num_vocab = len(vocab)
num_types_in_tokens = len(set(tokens))
word2id = {word: n for n, word in enumerate(vocab)}
token_ids = [word2id[w] for w in tokens]
print()
print('num_vocab={}'.format(num_vocab))
print('num types in tokens={}'.format(num_types_in_tokens))
if not num_types_in_tokens == num_vocab:
    print('Not all types ({}/{} were found in tokens.'.format(num_types_in_tokens, num_vocab))


# probes_data
num_cats2probes_data = {}
num_cats2max_ba = {}
for num_cats in params.num_cats_list:
    print('Getting {} categories with MIN_COUNT={}...'.format(num_cats, params.parent_count))
    legals_mat = ngram2legals_mat[params.structure_ngram_size]
    probes, probe2cat = make_probe_data(legals_mat, vocab, num_cats, params.parent_count,
                                        plot=False)
    num_cats2probes_data[num_cats] = (probes, probe2cat)
    c = Counter(tokens)
    for p in probes:
        # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bi-modality
        if c[p] < 10:
            print('WARNING: "{}" occurs only {} times'.format(p, c[p]))
    print('Collected {} probes'.format(len(probes)))
    # check probe sim
    probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
    ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
    probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
    ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
    print('input-data row-wise ba={:.3f}'.format(ba1))
    print('input-data col-wise ba={:.3f}'.format(ba2))
    print()
    num_cats2max_ba[num_cats] = ba2


    # TODO don't use a binary measure - for each context-probe pair, don't increment by 1 or zero (depending on diagnosticity of context)
    # TODO  numerator: increment by number of times context occurs with words that are members of category of current word
    # TODO denominator: increment by number of times context occurs with words that are NOT members of category of current word