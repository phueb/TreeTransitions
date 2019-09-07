import argparse
import pickle
import socket
from datetime import datetime
import yaml

from treetransitions import config
from treetransitions.job import main
from treetransitions.params import param2requests, param2default

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    config.Eval.plot_legal_mats = False
    config.Eval.plot_legal_corr_mats = False
    config.Eval.plot_tree = False

    p = config.RemoteDirs.root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:

        # check if host is down - do this before any computation
        assert config.RemoteDirs.runs.exists()  # this throws error if host is down

        # execute job
        main(param2val)

        # write param2val to shared drive
        param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
        if not param2val_p.exists():
            param2val_p['job_name'] = None
            with param2val_p.open('w', encoding='utf8') as f:
                yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)

    print('Finished all {} jobs at {}.'.format(config.LocalDirs.src.name, datetime.now()))
    print()


def run_on_host():
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals
    #
    for param2val in list_all_param2vals(param2requests, param2default,
                                         update_d={'param_name': 'test', 'job_name': 'test'}):
        if config.Eval.debug:
            param2val['num_seqs'] = 1 * 10 ** 3
            param2val['num_partitions'] = 2
            param2val['num_iterations'] = 1
            print('DEBUG=True: num_seqs={}'.format(param2val['num_seqs']))
        main(param2val)
        raise SystemExit('Finished running first job.\n'
                         'No further jobs will be run as results would be over-written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    #
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()