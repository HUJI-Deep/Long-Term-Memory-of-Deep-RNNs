import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.RNN_wrapper import *


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_conf_name(cell, depth, width, bs, optimizer, eta):
    return cell + "_depth" + str(depth) + "_width" + str(width) \
    + "_BS" + str(bs) + "_" + optimizer + "_eta%.4f" % eta
