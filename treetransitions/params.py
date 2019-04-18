

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input data
    truncate_num_cats = [32]  # TODO test
    truncate_list = [[0.5, 1.0], [1.0, 0.5]]
    num_tokens = [5 * 10 ** 6]
    max_ngram_size = [1]
    num_descendants = [2]  # 2
    num_levels = [10]  # 12
    mutation_prob = [0.2]  # 0.2
    parent_count = [1024]  # exact size of single parent cluster
    num_cats_list = [[32]]
    structure_ngram_size = [1]
    # rnn
    num_iterations = [20]
    num_partitions = [2]
    rnn_type =['srn']
    bptt = max_ngram_size
    mb_size = [64]
    learning_rate = [0.001]  # 0.01 is too fast
    num_hiddens = [128]
    optimization = ['adagrad']


class DefaultParams:
    # input data
    truncate_num_cats = [32]  # TODO test
    truncate_list = [[0.5, 1.0], [1.0, 0.5]]
    num_tokens = [5 * 10 ** 6]
    max_ngram_size = [1]
    num_descendants = [2]
    num_levels = [12]
    mutation_prob = [0.2]
    parent_count = [1024]  # exact size of single parent cluster
    num_cats_list = [[32]]
    structure_ngram_size = [1]
    # rnn
    num_iterations = [20]
    num_partitions = [2]
    rnn_type =['srn']
    bptt = max_ngram_size
    mb_size = [64]
    learning_rate = [0.001]  # 0.01 is too fast
    num_hiddens = [128]
    optimization = ['adagrad']