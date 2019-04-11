from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import yaml
import pandas as pd

from treetransitions.hierarchical_data_utils import make_data, make_probe_data, calc_ba
from treetransitions.rnn import RNN
from treetransitions.params import ObjectView
from treetransitions import config


def main_job(param2val, min_probe_freq=10):
    # params
    params = ObjectView(param2val.copy())

    # make tokens with hierarchical n-gram structure
    vocab, tokens, ngram2legals_mat = make_data(
        params.NUM_TOKENS, params.LEGALS_DISTRIBUTION, params.MAX_NGRAM_SIZE,
        params.NUM_DESCENDANTS, params.NUM_LEVELS, params.E, params.TRUNCATE)
    num_vocab = len(vocab)
    num_types_in_tokens = len(set(tokens))
    word2id = {word: n for n, word in enumerate(vocab)}
    token_ids = [word2id[w] for w in tokens]
    print()
    print('num_vocab={}'.format(num_vocab))
    print('num types in tokens={}'.format(num_types_in_tokens))
    if not num_types_in_tokens == num_vocab:
        raise RuntimeError('Not all types ({}/{} were found in tokens.'.format(
            num_types_in_tokens, num_vocab) + 'Decrease NUM_LEVELS, increase NUM_TOKENS, or increase TRUNCATE.')

    #
    num_theoretical_legals = num_vocab / (2 ** params.MAX_NGRAM_SIZE)
    print('num_theoretical_legals={}'.format(num_theoretical_legals))  # perplexity should converge to this value

    # train_seqs
    train_seqs = []
    for seq in itertoolz.partition_all(params.mb_size, token_ids):  # a seq contains mb_size token_ids
        if len(seq) == params.mb_size:
            train_seqs.append(list(seq))  # need to convert tuple to list
    print('num sequences={}'.format(len(train_seqs)))

    # probes_data
    num_cats2probes_data = {}
    num_cats2max_ba = {}
    for num_cats in params.NUM_CATS_LIST:
        print('Getting {} categories with MIN_COUNT={}...'.format(num_cats, params.PARENT_COUNT))
        legals_mat = ngram2legals_mat[params.NGRAM_SIZE_FOR_CAT]
        probes, probe2cat = make_probe_data(legals_mat, vocab, num_cats, params.PARENT_COUNT,
                                            plot=False)
        num_cats2probes_data[num_cats] = (probes, probe2cat)
        c = Counter(tokens)
        for p in probes:
            # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bi-modality
            if c[p] < min_probe_freq:
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


    # TODO add option to train on local iterations vs global iterations

    # train once
    srn = RNN(num_vocab, params)  # num_seqs_in_batch must be 1 to ensure mb_size is as specified in params
    lr = srn.learning_rate[0]  # initial
    decay = srn.learning_rate[1]
    num_epochs_without_decay = srn.learning_rate[2]
    num_cats2bas = {num_cats: [] for num_cats in params.NUM_CATS_LIST}
    for epoch in range(srn.num_epochs):
        # perplexity
        pp = srn.calc_seqs_pp(train_seqs[:params.num_pp_seqs])  # TODO how to calc pp on all seqs without mem error?
        # ba
        for num_cats, (probes, probe2cat) in sorted(num_cats2probes_data.items(), key=lambda i: i[0]):
            wx = srn.get_wx()  # TODO also test wy
            p_acts = np.asarray([wx[word2id[p], :] for p in probes])
            ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
            num_cats2bas[num_cats].append(ba)
            print('epoch={:>2}/{:>2} | ba={:.3f} num_cats={}'.format(epoch, srn.num_epochs, ba, num_cats))
        # train
        lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
        lr = lr * lr_decay  # decay lr if it is time
        srn.train_epoch(train_seqs, lr, verbose=False)
        print('epoch={:>2}/{:>2} | pp={:>5}\n'.format(epoch, srn.num_epochs, int(pp)))

    # traj_df
    traj_df = pd.DataFrame(num_cats2bas)  # TODO test

    print(traj_df)

    #  save traj_df to shared drive
    results_p = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name'] / 'results.csv'
    if not results_p.parent.exists():
        results_p.parent.mkdir(parents=True)
    with results_p.open('w') as f:  # TODO test
        traj_df.to_csv(f, index=False)

    # write param2val to shared drive
    param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)