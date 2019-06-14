class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input
    non_probes_hierarchy = [False]
    legal_probs = [[0.5, 1.0], [1.0, 1.0]]  # probability of legal sequence being counted as legal
    num_non_probes_list = [[512], [512, 512], [512, 512, 512]]  # there can be multiple non-probe categories
    num_probes = [512]
    num_contexts = [512]  # a smaller number reduces category structure of probes
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


for probs in Params.legal_probs:
    if probs[0] != probs[1]:
        for num_partitions in Params.num_partitions:
            assert num_partitions != 1  # no incremental structure without num_partitions > 1
