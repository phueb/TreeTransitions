

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input data
    NUM_TOKENS = [5 * 10 ** 6]
    MAX_NGRAM_SIZE = [1]
    NUM_DESCENDANTS = [2]  # 2
    NUM_LEVELS = [10]  # 12
    E = [0.2]  # 0.2
    TRUNCATE = [0.5, 0.75, 1.0]
    LEGALS_DISTRIBUTION = ['uniform', 'triangular']
    PARENT_COUNT = [1024]  # exact size of single parent cluster
    NUM_CATS_LIST = [[2, 4, 8, 16, 32]]
    NGRAM_SIZE_FOR_CAT = [1]  # TODO manipulate this - or concatenate all structures?
    # rnn
    rnn_type =['srn']
    bptt = MAX_NGRAM_SIZE
    mb_size = [64]
    learning_rate = [[0.001, 0.00, 20]]  # 0.01 is too fast
    num_epochs = [20]
    num_hiddens = [128]
    num_pp_seqs = [10]  # number of documents to calc perplexity for
    optimization = ['adagrad']


class DefaultParams:
    # input data
    NUM_TOKENS = [5 * 10 ** 6]
    MAX_NGRAM_SIZE = [1]
    NUM_DESCENDANTS = [2]
    NUM_LEVELS = [10]
    E = [0.2]
    TRUNCATE = [0.5, 0.75, 1.0]
    LEGALS_DISTRIBUTION = ['uniform']
    PARENT_COUNT = [256]  # exact size of single parent cluster
    NUM_CATS_LIST = [[2, 4, 8, 16, 32]]
    NGRAM_SIZE_FOR_CAT = [1]
    # rnn
    rnn_type =['srn']
    bptt = MAX_NGRAM_SIZE
    mb_size = [64]
    learning_rate = [[0.001, 0.00, 20]]  # 0.01 is too fast
    num_epochs = [20]
    num_hiddens = [128]
    num_pp_seqs = [10]  # number of documents to calc perplexity for
    optimization = ['adagrad']