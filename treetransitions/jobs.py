from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import yaml
import pandas as pd
import sys

from treetransitions.hierarchical_data_utils import make_tokens, make_probe_data, calc_ba, make_legal_mats, make_vocab
from treetransitions.rnn import RNN
from treetransitions.params import ObjectView
from treetransitions import config

TRUNCATE_SIZE = 1


def generate_toy_data(params, num_cats):
    toy_data = {}

    vocab, word2id = make_vocab(params.num_descendants, params.num_levels)

    # make underlying hierarchical structure
    size2word2legals, ngram2legals_mat = make_legal_mats(
        vocab, params.num_descendants, params.num_levels, params.mutation_prob, params.max_ngram_size)

    # probes_data
    num_cats2word2sorted_legals = {}
    print('Getting {} categories with parent_count={}...'.format(num_cats, params.parent_count))
    legals_mat = ngram2legals_mat[params.structure_ngram_size]
    probes, probe2cat, word2sorted_legals = make_probe_data(
        vocab, size2word2legals[TRUNCATE_SIZE], legals_mat, num_cats, params.parent_count, params.truncate_control)
    print('Collected {} probes'.format(len(probes)))
    # check probe sim
    probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
    ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
    probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
    ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
    print('input-data row-wise ba={:.3f}'.format(ba1))
    print('input-data col-wise ba={:.3f}'.format(ba2))
    print()
    num_cats2word2sorted_legals[num_cats] = word2sorted_legals

    # sample tokens
    tokens = make_tokens(vocab, size2word2legals, num_cats2word2sorted_legals[params.truncate_num_cats],
                         params.num_tokens, params.max_ngram_size, params.truncate_list)

    toy_data['tokens'] = tokens
    toy_data['vocab'] = vocab
    toy_data['probe2cat'] = probe2cat
    res = ObjectView(toy_data)
    return res


def main_job(param2val, min_probe_freq=10):
    # check if host is down - do this before any computation
    results_p = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name'] / 'results.csv'
    assert config.RemoteDirs.runs.exists()    # this throws error if host is down

    # params
    params = ObjectView(param2val.copy())
    for k, v in param2val.items():
        print('{}={}'.format(k, v))
    print()

    vocab, word2id = make_vocab(params.num_descendants, params.num_levels)

    # make underlying hierarchical structure
    size2word2legals, ngram2legals_mat = make_legal_mats(
        vocab, params.num_descendants, params.num_levels, params.mutation_prob, params.max_ngram_size)

    # probes_data
    num_cats2probes_data = {}
    num_cats2max_ba = {}
    num_cats2word2sorted_legals = {}
    for num_cats in params.num_cats_list:
        print('Getting {} categories with parent_count={}...'.format(num_cats, params.parent_count))
        legals_mat = ngram2legals_mat[params.structure_ngram_size]
        probes, probe2cat, word2sorted_legals = make_probe_data(
            vocab, size2word2legals[TRUNCATE_SIZE], legals_mat, num_cats, params.parent_count, params.truncate_control)
        num_cats2probes_data[num_cats] = (probes, probe2cat)
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
        num_cats2word2sorted_legals[num_cats] = word2sorted_legals

    # sample tokens
    tokens = make_tokens(vocab, size2word2legals, num_cats2word2sorted_legals[params.truncate_num_cats],
                         params.num_tokens, params.max_ngram_size, params.truncate_list)
    num_vocab = len(vocab)
    num_types_in_tokens = len(set(tokens))
    word2id = {word: n for n, word in enumerate(vocab)}
    token_ids = [word2id[w] for w in tokens]
    print()
    print('num_vocab={}'.format(num_vocab))
    print('num types in tokens={}'.format(num_types_in_tokens))
    if not num_types_in_tokens == num_vocab:
        print('Not all types ({}/{} were found in tokens.'.format(num_types_in_tokens, num_vocab))
    num_theoretical_legals = num_vocab / (2 ** params.max_ngram_size)
    print('num_theoretical_legals={}'.format(num_theoretical_legals))  # perplexity should converge to this value

    # check probe frequency
    c = Counter(tokens)
    for num_cats, (probes, probe2cat) in num_cats2probes_data.items():
        for p in probes:
            # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bi-modality
            if c[p] < min_probe_freq:
                print('WARNING: "{}" occurs only {} times'.format(p, c[p]))

    # train loop
    srn = RNN(num_vocab, params)  # num_seqs_in_batch must be 1 to ensure mb_size is as specified in params
    num_cats2bas = {num_cats: [] for num_cats in params.num_cats_list}
    part_size = params.num_tokens // params.num_partitions
    part_id = 0
    for part in itertoolz.partition_all(part_size, token_ids):
        if len(part) != part_size:
            continue
        part_id += 1
        seqs_in_part = [list(seq) for seq in itertoolz.partition_all(params.mb_size, part)]
        print('num mb_size sequences in partition={}'.format(len(seqs_in_part)))
        # perplexity
        pp = srn.calc_seqs_pp(seqs_in_part) if config.Eval.calc_pp else 0
        # iterations
        for iteration in range(params.num_iterations):
            # ba
            for num_cats, (probes, probe2cat) in sorted(num_cats2probes_data.items(), key=lambda i: i[0]):
                wx = srn.get_wx()  # TODO also test wy
                p_acts = np.asarray([wx[word2id[p], :] for p in probes])
                ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
                num_cats2bas[num_cats].append(ba)
                print('partition={:>2}/{:>2} | ba={:.3f} num_cats={}'.format(part_id, srn.num_partitions, ba, num_cats))
            #
            print('partition={:>2}/{:>2} | before-training partition pp={:>5}\n'.format(
                part_id, srn.num_partitions, pp))
            sys.stdout.flush()
            # train
            srn.train_partition(seqs_in_part, verbose=False)  # a seq is a list of mb_size token_ids

    #  save results to disk
    if not results_p.parent.exists():
        results_p.parent.mkdir(parents=True)
    traj_df = pd.DataFrame(num_cats2bas)
    with results_p.open('w') as f:
        traj_df.to_csv(f, index=False)

    # write param2val to shared drive
    param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)