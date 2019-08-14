def conf_name(M, n, logspace, cell, depth, width, batch_size, optimizer, eta, hidden_eta, alpha):
    return "M" + str(M) + "_n" + str(n) + "_" + ("logspace-" if logspace else "") + \
                cell + "_depth" + str(depth) + "_width" + str(width) + \
                "_BS" + str(batch_size) + "_" + optimizer + "_eta%.4f" % eta + "_hiddeneta%.4f" % hidden_eta + (
                "_alpha%.3f" % alpha if alpha > 0 else "")