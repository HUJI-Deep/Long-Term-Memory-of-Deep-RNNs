import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.RNN_wrapper import *
import time

BLANK = '-'
START_RECALL = ':'
A_ASCII = 65
ASCII_MAX = 126
special_chars = [BLANK, START_RECALL]
args = None
alphabet = None
char_to_i = None
floattype = None
T = -1
confname = ""


def conf_name(args):
    return "B" + str(args.B) + "_" + "M" + str(args.M) + "_n" + str(args.n) + \
                '_' + args.rnn_cell + "_depth" + str(args.rnn_depth) + "_width" + str(args.rnn_hidden_dim) + \
                "_BS" + str(args.batch_size) + "_" + args.optimizer + "_eta%.4f" % args.learning_rate


def set_globals(args_, alphabet_, char_to_i_, floattype_, confname_):
    global args, alphabet, char_to_i, floattype, T, confname
    args = args_
    alphabet = alphabet_
    char_to_i = char_to_i_
    floattype = floattype_
    T = 2 * args.M + args.B
    confname = confname_


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def one_hots_to_str(X):
    return [''.join(alphabet[np.argmax(X[i], axis=1)]) for i in range(X.shape[0])]


def idx_to_str(y):
    return [''.join(alphabet[y[i]]) for i in range(y.shape[0])]


def to_one_hot(x, tf_session):
    # convert x to one_hot
    x_ = tf.placeholder("int32", [dim for dim in x.shape])
    X_tensor = tf.one_hot(x, len(alphabet), dtype=floattype)
    return tf_session.run(X_tensor, feed_dict={x_: x})


def preprocess_data(words):
    num_samples = words.shape[0]
    assert words.shape[1] == args.M
    x = np.column_stack((words, np.full((num_samples, args.B + args.M), char_to_i[BLANK]))).astype(np.uint32)
    x[:, T - args.M] = np.full((num_samples,), char_to_i[START_RECALL])
    y = np.column_stack((np.full((num_samples, args.M + args.B), char_to_i[BLANK]), words)).astype(np.int32)
    X = x
    return X, y


def generate_batch(num_examples):
    words = np.random.randint(args.n, size=(num_examples, args.M))
    X, y = preprocess_data(words)
    return X, y


def get_data(test_size, validation_size):
    data_dir = 'data/'
    make_sure_path_exists(data_dir)
    data_filename = data_dir + 'M' + str(args.M) + '_n' + str(args.n) + '_' + 'BS' + str(args.batch_size) + '_data.npz'
    try:
        data = np.load(data_filename if not args.generate_data else '')
    except FileNotFoundError:
        test_words = np.random.randint(args.n, size=(test_size, args.M))
        validation_words = np.random.randint(args.n, size=(validation_size, args.M))
        np.savez(data_filename, validation_words=validation_words, test_words=test_words)
    else:
        test_words = data['test_words']
        validation_words = data['validation_words']

    X_test, y_test = preprocess_data(test_words)
    X_validation, y_validation = preprocess_data(validation_words)
    return X_test, y_test, X_validation, y_validation


def new_rnn(tf_session):
    rnn = RNN(len(alphabet), len(alphabet), args.rnn_hidden_dim, T, args.rnn_depth, args.batch_size,
              tf_session, confname, cell_name=args.rnn_cell, to_one_hot=True, learning_rate=args.learning_rate,
              optimizer_name=args.optimizer, tb_verbosity=1, log_period=100, print_verbosity=args.print_verbosity)
    return rnn


def run_train(rnn, X_validation, y_validation):
    start_time = time.time()

    early_stop_TH = 1e-3 * (args.M / T)
    if args.print_verbosity > 1:
        print("early_stop_TH:", early_stop_TH)
    epochs_till_converge = rnn.train(generate_batch, args.num_iters, X_validation, y_validation,
                                     load_weights=args.load_weights, auto_learning_rate_decay=True,
                                     convergence_min_delta=early_stop_TH, convergence_patience=10)
    runtime = time.time() - start_time

    return epochs_till_converge, runtime


def train_models(X_validation, y_validation):
    tf_session = tf.Session()
    rnn = new_rnn(tf_session)
    tf_session.run(tf.global_variables_initializer())
    epochs_to_converge, runtime = run_train(rnn, X_validation, y_validation)

    if args.print_verbosity > 0:
        print("num_epochs: %d  runtime: %d" % (epochs_to_converge, runtime))

    return rnn, tf_session, epochs_to_converge, runtime
