import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.RNN import *


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_conf_name(cell, depth, width, bs, optimizer, eta, hidden_eta, rms_decay):
    return cell + "_depth" + str(depth) + "_width" + str(width) \
    + "_BS" + str(bs) + "_" + optimizer + "_eta%.4f" % eta + "_hiddeneta%.4f" % hidden_eta + "_rms_decay%.1f" % rms_decay