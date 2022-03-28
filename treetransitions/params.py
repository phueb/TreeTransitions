from typing import Tuple
from dataclasses import dataclass


param2requests = {



}


param2default = {

    'num_cats_list': (2, 4, 8, 16, 32),
    'num_iterations': 20,
    'num_partitions': 2,
    'rnn_type': 'srn',
    'batch_size': 64,
    'learning_rate': 0.03,  # 0.03-adagrad 0.3-sgd
    'num_hidden': 128,
    'optimization': 'adagrad',  # don't forget to change learning rate
}


@dataclass
class Params(object):
    num_cats_list: Tuple[int, ...]
    num_iterations: int
    num_partitions: int
    rnn_type: str
    batch_size: int
    learning_rate: float
    num_hidden: int
    optimization: str

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)
