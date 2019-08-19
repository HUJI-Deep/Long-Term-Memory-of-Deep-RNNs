import os, sys, errno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from RNN_wrapper import *

n = -1
C = -1
m = -1
alphabet = None
char_to_i = None

def set_globals(n_, C_, args_, m_, alphabet_, char_to_i_):
    global n, C, args, m, alphabet, char_to_i
    n = n_
    args = args_
    C = min(C_, args.T // 2)
    m = m_
    alphabet = alphabet_
    char_to_i = char_to_i_
    assert args.T % 2 == 0, "ERROR: start-end similarity doesn't support odd T values"


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_conf_name(C, n, m, cell, depth, width, bs, optimizer, eta):
    return "C" + str(C) + "_n" + str(n) + "_m" + str(m) + '_' + cell + "_depth" + str(depth) + "_width" + str(width) \
    + "_BS" + str(bs) + "_" + optimizer + "_eta%.4f" % eta


def idx_to_str(y):
    return [''.join(alphabet[y[i, :]]) for i in range(y.shape[0])]


def calc_y(X):
    def get_scores(X):
        # m = T//2
        pos = np.argmax(X < n, axis=1)
        sample_idx = np.matlib.repmat(np.arange(X.shape[0]), m, 1).T
        pos_idx = np.matlib.repmat(pos, m, 1).T + np.matlib.repmat(np.arange(m), X.shape[0], 1)
        first_half = X[sample_idx, pos_idx]
        second_half = X[sample_idx, pos_idx + args.T // 2]
        refl_counts = np.sum(np.equal(first_half, second_half), axis=1)
        return refl_counts / m

    scores = get_scores(X)
    y = np.round(C * scores)
    y = np.expand_dims(y, axis=1)
    return y


def uniform_y_data(data_size):
    # m = T//2
    num_overlaps = np.round(np.random.randint(C+1, size=data_size) * float(m / C)).astype(np.int32)
    half_sequences = np.random.randint(n, size=(data_size, m))
    second_half = np.full((data_size, m), -1)
    for i, seq in enumerate(half_sequences):
        overlap_indices = random.sample(range(m), num_overlaps[i])
        for t in range(m):
            if t in overlap_indices:
                second_half[i, t] = seq[t]
            else:
                non_refl_vals = np.concatenate((np.arange(seq[t]), np.arange(seq[t] + 1, n)))
                second_half[i, t] = np.random.choice(non_refl_vals)

    if m < args.T//2:
        pos = np.random.randint(args.T//2-m, size=data_size)
        X = np.full((data_size, args.T), n)
        sample_idx = np.matlib.repmat(np.arange(data_size), m, 1).T
        pos_idx = np.matlib.repmat(pos, m, 1).T + np.matlib.repmat(np.arange(m), data_size, 1)
        X[sample_idx, pos_idx] = half_sequences
        X[sample_idx, pos_idx+args.T//2] = second_half
        return X
    else:
        return np.concatenate((half_sequences, second_half), axis=1)


def sample_data(num_samples):
    X = uniform_y_data(num_samples)
    y = calc_y(X)
    return X, y


def get_data(test_size, validation_size):
    make_sure_path_exists('data')
    data_filename = 'data/' + 'data_T' + str(args.T) + '_C' + str(C) + '_n' + str(n) + '_m' + str(m) + '_bs' + str(args.batch_size) + '.npz'
    try:
        data = np.load(data_filename if not args.generate_data else '')
    except FileNotFoundError:
        X_test, y_test = sample_data(test_size)
        X_validation, y_validation = sample_data(validation_size)
        np.savez(data_filename, X_test=X_test, y_test=y_test, X_validation=X_validation, y_validation=y_validation)
    else:
        X_test = data['X_test']
        y_test = data['y_test']
        X_validation = data['X_validation']
        y_validation = data['y_validation']

    assert X_test.shape[0] == test_size
    assert X_validation.shape[0] == validation_size
    return X_test, y_test, X_validation, y_validation


def train_rnn(X_validation, y_validation, conf_name):
    def new_rnn(seed=-1):
        rnn = RNN(len(alphabet), (C+1), args.rnn_hidden_dim, args.T, args.rnn_depth, args.batch_size, tf_session,
                  "T" + str(args.T) + "_" + conf_name + ("_seed" + str(seed) if not args.load_weights else ""),
                  args.rnn_cell, single_output=True, to_one_hot=True,
                  learning_rate=args.learning_rate,
                  optimizer_name=args.optimizer,
                  tb_verbosity=args.tb_verbosity, print_verbosity=args.print_verbosity)
        return rnn

    weights_dir = 'rnn_weights'
    weights_file = weights_dir + '/' + "T" + str(args.T) + "_" + conf_name + '.ckpt'

    # set session and seed
    seed = np.random.randint(0, 10000)
    tf_session = tf.Session()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # train
    rnn = new_rnn(seed)
    tf_session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if args.load_weights:
        saver.restore(tf_session, weights_file)
        if args.print_verbosity > 1:
            print("WEIGHTS LOADED SUCCESSFULLY!")

    convergence_min_delta = 1e-3
    if args.print_verbosity > 1:
        print("min_delta %f" % convergence_min_delta)
    num_iterations = rnn.train(sample_data, args.num_iters, X_validation, y_validation, auto_learning_rate_decay=True,
                               convergence_min_delta=convergence_min_delta, convergence_patience=10)

    return rnn, num_iterations, tf_session
