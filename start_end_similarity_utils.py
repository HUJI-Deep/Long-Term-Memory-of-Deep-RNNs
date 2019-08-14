import os, sys, errno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import pickle
import numpy.matlib
from common.RNN import *

n = -1
C = -1
T = -1
m = -1
alphabet = None
char_to_i = None
trigger_char = False
random_pos = False
fully_rand_pos = False
num_iters = -1
rnn_num_retrainings = -1
params = None
load_weights = False
generate_data = -1
print_verbosity = -1
tb_verbosity = -1
unit_testing = -1
metadata_filename = ""
dump_metadata = True
chosen_seed = -1


def set_globals(n_, C_, T_, m_, alphabet_, char_to_i_, trigger_char_, random_pos_, fully_rand_pos_, num_iters_,
                rnn_num_retrainings_, params_, load_weights_, generate_data_, print_verbosity_, tb_verbosity_,
                unit_testing_, metadata_filename_, chosen_seed_):
    global n, C, T, m, alphabet, char_to_i, trigger_char, random_pos, fully_rand_pos, num_iters, rnn_num_retrainings, params, \
        load_weights, generate_data, print_verbosity, tb_verbosity, unit_testing, metadata_filename, chosen_seed
    n = n_
    T = T_
    C = min(C_, T // 2)
    m = m_
    alphabet = alphabet_
    char_to_i = char_to_i_
    trigger_char = trigger_char_
    random_pos = random_pos_
    fully_rand_pos = fully_rand_pos_
    num_iters = num_iters_
    rnn_num_retrainings = rnn_num_retrainings_
    params = params_
    load_weights = load_weights_
    generate_data = generate_data_
    print_verbosity = print_verbosity_
    tb_verbosity = tb_verbosity_
    unit_testing = unit_testing_
    metadata_filename = metadata_filename_
    chosen_seed = chosen_seed_

    assert T % 2 == 0, "ERROR: start-end similarity doesn't support odd T values"


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_conf_name(C, n, m, cell, depth, width, bs, optimizer, eta, hidden_eta, rms_decay):
    return "C" + str(C) + "_n" + str(n) + "_m" + str(m) + '_' + cell + "_depth" + str(depth) + "_width" + str(width) \
    + "_BS" + str(bs) + "_" + optimizer + "_eta%.4f" % eta + "_hiddeneta%.4f" % hidden_eta + "_rms_decay%.1f" % rms_decay


def idx_to_str(y):
    return [''.join(alphabet[y[i, :]]) for i in range(y.shape[0])]


def calc_y(X):
    def get_scores(X):
        # m = T//2
        if fully_rand_pos:
            X_reduced = np.reshape(X[X < n], (X.shape[0], 2*m))
            first_half = X_reduced[:, :m]
            second_half = X_reduced[:, m:]
        elif random_pos:
            pos = np.argmax(X < n, axis=1)
            sample_idx = np.matlib.repmat(np.arange(X.shape[0]), m, 1).T
            pos_idx = np.matlib.repmat(pos, m, 1).T + np.matlib.repmat(np.arange(m), X.shape[0], 1)
            first_half = X[sample_idx, pos_idx]
            second_half = X[sample_idx, pos_idx + T // 2]
        else:
            first_half = X[:, :m]
            second_half = X[:, -m:]
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

    if trigger_char:
        # trigger char index is n
        return np.concatenate((half_sequences, np.matlib.repmat(np.array([n]), data_size, 1), second_half), axis=1)
    elif m < T//2:
        # blank char index is n
        if fully_rand_pos:
            X = np.full((data_size, T), n)
            for i in range(data_size):
                pos_start = np.sort(np.random.choice(np.arange(T//2), m, replace=False))
                pos_end = np.sort(np.random.choice(np.arange(T//2), m, replace=False) + T//2)
                X[i, pos_start] = half_sequences[i, :]
                X[i, pos_end] = second_half[i, :]
            return X
        elif random_pos:
            pos = np.random.randint(T//2-m, size=data_size)
            X = np.full((data_size, T), n)
            sample_idx = np.matlib.repmat(np.arange(data_size), m, 1).T
            pos_idx = np.matlib.repmat(pos, m, 1).T + np.matlib.repmat(np.arange(m), data_size, 1)
            X[sample_idx, pos_idx] = half_sequences
            X[sample_idx, pos_idx+T//2] = second_half
            return X
        else:
            return np.concatenate((half_sequences, np.full((data_size, T - 2*m), n), second_half), axis=1)
            # return np.concatenate((half_sequences, np.matlib.repmat(np.array([n]), data_size, T - 2*m), second_half), axis=1)
    else:
        return np.concatenate((half_sequences, second_half), axis=1)


def sample_data(num_samples):
    X = uniform_y_data(num_samples)
    y = calc_y(X)
    return X, y


def get_data(test_size, validation_size):
    make_sure_path_exists('data')
    data_filename = 'data/' + 'data_T' + str(T) + '_C' + str(C) + '_n' + str(n) + '_m' + str(m) + '_bs' + str(params['rnn_batch_size']) + '.npz'
    try:
        data = np.load(data_filename if not generate_data else '')
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


def dump_meta_data():
    global dump_metadata
    if dump_metadata:
        num_trainable_weights = sum(
            [np.prod(np.array([dim.value for dim in v.shape.dims])) for v in tf.trainable_variables()])
        make_sure_path_exists('outputs')
        with open(metadata_filename, 'wb') as f:
            pickle.dump([params, num_trainable_weights], f, pickle.HIGHEST_PROTOCOL)
        if print_verbosity > 0:
            print(params)
            print("num of rnn trainable weights: ", num_trainable_weights)
        dump_metadata = False


def train_rnn(X_validation, y_validation, conf_name):
    def new_rnn(seed=-1):
        rnn = RNN(len(alphabet), (C+1), params['rnn_hidden_dim'], T+(1 if trigger_char else 0), params['rnn_depth'], params['rnn_batch_size'], tf_session,
                  "T" + str(T) + "_" + conf_name + ("_seed" + str(seed) if not load_weights else ""),
                  params['rnn_cell'], single_output=True, logspace=False, to_one_hot=True,
                  learning_rate=params['rnn_learning_rate'],
                  hidden_eta=params['rnn_hidden_eta'],
                  optimizer_name=params['optimizer'], rms_decay=params['rms_decay'],
                  tb_verbosity=tb_verbosity, print_verbosity=print_verbosity)
        return rnn

    def single_train(seed):
        rnn = new_rnn(seed)
        tf_session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if load_weights:
            saver.restore(tf_session, weights_file)
            if print_verbosity > 1:
                print("WEIGHTS LOADED SUCCESSFULLY!")

        dump_meta_data()
        convergence_min_delta = 1e-3
        if print_verbosity > 1:
            print("min_delta %f" % convergence_min_delta)
        num_iterations = rnn.train(sample_data, num_iters, X_validation, y_validation, auto_learning_rate_decay=True,
                               convergence_min_delta=convergence_min_delta, convergence_patience=10)
        return rnn, num_iterations, saver

    # np.random.seed(0)
    if chosen_seed == -1:
        seeds = np.random.randint(0, 10000, rnn_num_retrainings)
    else:
        assert rnn_num_retrainings == 1, "manual seed is available for single retraining mode only"
        seeds = [chosen_seed]
    best_loss = np.inf
    num_iters_best_seed = -1
    weights_dir = 'rnn_weights'
    weights_file = weights_dir + '/' + "T" + str(T) + "_" + conf_name + '.ckpt'
    for seed in seeds:
        # set session and seed
        tf_session = tf.Session()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        # train
        rnn, num_iterations, saver = single_train(seed)
        # evaluate and save if best
        _, o = rnn.predict(X_validation)
        validation_loss = tf_session.run(rnn.loss_by_ph, feed_dict={rnn.logits_ph_: o, rnn.y_: y_validation})
        if print_verbosity > 1:
            print("seed: %d  validation_loss: %f" % (seed, validation_loss))
        if validation_loss < best_loss:
            if print_verbosity > 1:
                print("SAVING WEIGHTS!" % validation_loss)
            best_loss = validation_loss
            num_iters_best_seed = num_iterations
            best_test_o = o
            make_sure_path_exists(weights_dir)
            saver.save(tf_session, weights_file)


        # reset
        tf.reset_default_graph()
        tf_session.close()

    tf_session = tf.Session()
    rnn = new_rnn()
    final_saver = tf.train.Saver()
    final_saver.restore(tf_session, weights_file)
    if print_verbosity > 1:
        print("WEIGHTS LOADED SUCCESSFULLY!")

    return rnn, best_test_o, num_iters_best_seed, tf_session
