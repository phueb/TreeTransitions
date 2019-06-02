class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input
    legals_mat_seed = [42]  # TODO test
    corpus_seed = [42]  # TODO test - do i also need a seed for branching diffusion?
    non_probes_hierarchy = [False]
    structure_probs = [[1.0, 1.0], [0.5, 1.0]]  # probability of drawing a 1 from [1, -1]; a 1 preserves hierarchy
    num_non_probes_list = [[1024, 1024]]  # there can be multiple non-probe categories
    num_probes = [512]
    num_contexts = [128]
    num_seqs = [5 * 10 ** 6]
    mutation_prob = [0.01]
    num_cats_list = [[2, 4, 8, 16, 32]]
    # rnn
    num_iterations = [20]
    num_partitions = [2]
    rnn_type = ['srn']
    mb_size = [64]
    learning_rate = [0.03]  # 0.03-adagrad 0.3-sgd
    num_hiddens = [128]
    optimization = ['adagrad']  # don't forget to change learning rate
    # eval
    w = ['embeds']


for probs in Params.structure_probs:
    if probs[0] != probs[1]:
        for num_partitions in Params.num_partitions:
            assert num_partitions != 1  # no incremental structure without num_partitions > 1
