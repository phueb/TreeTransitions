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



class Graph:
    device = 'gpu'


class Figs:
    lw = 2
    axlabel_fs = 16
    leg_fs = 16
    dpi = None