def conf_name(args):
    return "M" + str(args.M) + "_n" + str(args.n) + \
                args.rnn_cell + "_depth" + str(args.rnn_depth) + "_width" + str(args.rnn_hidden_dim) + \
                "_BS" + str(args.batch_size) + "_" + args.optimizer + "_eta%.4f" % args.learning_rate