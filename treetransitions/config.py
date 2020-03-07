import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dirs:
    root = Path(__file__).parent.parent
    src = root / 'treetransitions'


class Eval:
    debug = False
    calc_pp = False
    num_processes = 4


class Graph:
    device = 'gpu'


class Seed:
    branching_diffusion = 10
    legals = 11
    sampling = 12