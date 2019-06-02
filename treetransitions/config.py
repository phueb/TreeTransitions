import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RemoteDirs:
    root = Path('/media/lab') / 'TreeTransitions'
    runs = root / 'runs'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'treetransitions'
    runs = root / '{}_runs'.format(src.name)


class Eval:
    debug = False
    calc_pp = False
    num_processes = 4
    plot_tree = False
    plot_legals_mat = False
    plot_corr_mat = False


class Graph:
    device = 'gpu'


class Seed:
    branching_diffusion = 10
    legals = 11
    sampling = 12