from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import yaml
import pandas as pd
import sys

from treetransitions.toy_data import ToyData
from treetransitions.utils import calc_ba
from treetransitions.rnn import RNN
from treetransitions.params import ObjectView
from treetransitions import config

TRUNCATE_SIZE = 1


def main_job(param2val, min_probe_freq=10):
    # check if host is down - do this before any computation
    results_p = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name'] / 'results.csv'
    assert config.RemoteDirs.runs.exists()    # this throws error if host is down

    # params
    params = ObjectView(param2val.copy())
    for k, v in param2val.items():
        print('{}={}'.format(k, v))
    print()

    toy_data = ToyData(params)

    # check probe frequency
    c = Counter(toy_data.tokens)
    for num_cats in params.num_cats_list:
        for p in toy_data.probes:
            # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bi-modality
            if c[p] < min_probe_freq:
                print('WARNING: "{}" occurs only {} times'.format(p, c[p]))

    # train loop
    num_vocab = len(toy_data.vocab)
    srn = RNN(num_vocab, params)  # num_seqs_in_batch must be 1 to ensure mb_size is as specified in params
    num_cats2bas = {num_cats: [] for num_cats in params.num_cats_list}
    part_size = params.num_tokens // params.num_partitions
    part_id = 0
    for part in itertoolz.partition_all(part_size, toy_data.token_ids):
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
            for num_cats in params.num_cats_list:
                wx = srn.get_wx()  # TODO also test wy
                #
                probes = toy_data.probes
                probe2cat = toy_data.num_cats2probe2cat[num_cats]
                p_acts = np.asarray([wx[toy_data.word2id[p], :] for p in probes])
                ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
                num_cats2bas[num_cats].append(ba)
                print('partition={:>2}/{:>2} | ba={:.3f} num_cats={}'.format(part_id, srn.num_partitions, ba, num_cats))
            #
            print('partition={:>2}/{:>2} iteration {}/{} | before-training partition pp={:>5}\n'.format(
                part_id, srn.num_partitions, iteration, params.num_iterations, pp))
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