import os, sys, errno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.RNN import *
from copy_task_conf_file import *
from copy_task_utils import *
import pprint
import argparse
import fcntl
import time
import pickle

parser = argparse.ArgumentParser(description='Script to run copy task with different rnn architectures')
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-test_size', type=int, help='test set size', required=True)
parser.add_argument('-validation_size', type=int, help='validation set size', required=True)
parser.add_argument('-rnn_num_retrainings', type=int, help='num retrainings of rnn', default=1)
parser.add_argument('-run_index', type=int, help='slurm run index', required=True)
parser.add_argument('-generate_data', type=bool, help='regenerate data')
parser.add_argument('-load_weights', type=bool, help='load existing weights at the beginning of train', default=0)
parser.add_argument('-to_one_hot_on_the_fly', type=bool, help='transform to one hot in computation graph', default=1)
parser.add_argument('-write_delay_time', type=bool, help='write delay time to tensorboard iff true otherwise write M', default=1)
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=0)

args = parser.parse_args()


args.write_delay_time = False

params = get_conf(args.run_index)
M = params['M']
BLANK = '-'
START_RECALL = ':'
A_ASCII = 65
ASCII_MAX = 126
special_chars = [BLANK, START_RECALL]
n = params['n']  # alphabet size
# assert n <= (ASCII_MAX - A_ASCII + 1)
alphabet = np.array([chr(x) for x in range(A_ASCII, A_ASCII + n)] + special_chars)
char_to_i = {c: i for (i, c) in enumerate(alphabet)}
floattype = tf.float64 if params['rnn_cell'] == 'OrthogonalRNN' else tf.float32
#np.random.seed(0)

confname = conf_name(params['M'], params['n'], params['logspace'], params['rnn_cell'] + ("_full_capacity" if params['rnn_cell'] == 'EUNN' and params['EUNN_full_capacity'] else ""),
                     params['rnn_depth'], params['rnn_hidden_dim'], params['rnn_batch_size'], params['optimizer'],
                     params['rnn_learning_rate'], params['rnn_hidden_eta'], params['regularization_alpha'])
tb_writer = tf.summary.FileWriter('tb_general/' + confname)
dump_metadata = True


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


class RandomRecall:
    def __init__(self, M, B, tf_session):
        self.M = M
        self.B = B
        self.T = 2*M + B
        self.tf_session = tf_session

    def train(self, X, y):
        pass

    def predict(self, X):
        num_samples = X.shape[0]
        rand_o = np.concatenate((np.full((num_samples, M, n), 1 / n), np.zeros((num_samples, M, len(special_chars)))), axis=2)
        rand_o = np.log(rand_o + np.random.uniform(1e-9, 1e-8, rand_o.shape))
        blanks = to_one_hot(np.full((num_samples, self.M + self.B), char_to_i[BLANK]), self.tf_session)
        blanks = np.log(blanks + np.full(blanks.shape, 1e-8))
        return np.concatenate((blanks, rand_o), axis=1)


def train_and_evaluate(B):
    T = 2 * M + B  # input/output dimension
    weights_dir = 'rnn_weights'
    weights_file = weights_dir + '/' + 'B' + str(B) + '_' + confname + '.pkl'
    if not args.to_one_hot_on_the_fly:
        preprocessing_tf_session = tf.Session()

    def preprocess_data(words):
        num_samples = words.shape[0]
        assert words.shape[1] == M
        x = np.column_stack((words, np.full((num_samples, B + M), char_to_i[BLANK]))).astype(np.uint32)
        x[:, T - M] = np.full((num_samples,), char_to_i[START_RECALL])
        y = np.column_stack((np.full((num_samples, M + B), char_to_i[BLANK]), words)).astype(np.int32)
        if args.to_one_hot_on_the_fly:
            X = x
        else:
            X = to_one_hot(x, preprocessing_tf_session)
        if params['logspace']:
            X_logspace = np.full(X.shape, -1e50)
            X_logspace[X.astype(np.bool)] = 0
            X = X_logspace
        return X, y

    def generate_batch(num_examples):
        words = np.random.randint(n, size=(num_examples, M))
        X, y = preprocess_data(words)
        return X, y

    def get_data(test_size, validation_size):
        data_filename = 'M' + str(params['M']) + '_n' + str(params['n']) + '_' + 'BS' + str(params['rnn_batch_size']) + '_data.npz'
        try:
            data = np.load(data_filename if not args.generate_data else '')
        except FileNotFoundError:
            test_words = np.random.randint(n, size=(test_size, M))
            validation_words = np.random.randint(n, size=(validation_size, M))
            f_stream = open(data_filename, 'w+')
            while True:
                try:
                    fcntl.flock(f_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            np.savez(data_filename, validation_words=validation_words, test_words=test_words)
            fcntl.flock(f_stream, fcntl.LOCK_UN)
        else:
            test_words = data['test_words']
            validation_words = data['validation_words']

        X_test, y_test = preprocess_data(test_words)
        X_validation, y_validation = preprocess_data(validation_words)
        return X_test, y_test, X_validation, y_validation

    # data preprocessing
    batch_size = params['rnn_batch_size']
    validation_size = (args.validation_size // batch_size) * batch_size
    test_size = (args.test_size // batch_size) * batch_size
    X_test, y_test, X_validation, y_validation = get_data(test_size, validation_size)

    def new_rnn(seed, tf_session):
        extended_conf_name = "B" + str(B) + "_" + confname + "_seed" + str(seed)
        rnn = RNN(len(alphabet), len(alphabet), params['rnn_hidden_dim'], T, params['rnn_depth'], batch_size,
                  tf_session,
                  extended_conf_name, cell_name=params['rnn_cell'], logspace=params['logspace'],
                  to_one_hot=args.to_one_hot_on_the_fly,
                  learning_rate=params['rnn_learning_rate'],
                  hidden_eta=params['rnn_hidden_eta'],
                  optimizer_name=params['optimizer'],
                  EUNN_full_capacity=params['EUNN_full_capacity'],
                  tb_verbosity=1, log_period=100,
                  print_verbosity=args.print_verbosity)
        return rnn

    def train_models(seed, best_loss):
        start_time = time.time()
        tf_session = tf.Session()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        y_ = tf.placeholder(tf.int64, shape=[None, T])
        o_ = tf.placeholder(floattype, shape=[None, T, len(alphabet)])
        rnn = new_rnn(seed, tf_session)
        tf_session.run(tf.global_variables_initializer())

        if args.load_weights:
            saved_vars = pickle.load(open(weights_file, 'rb'))
            tf_session.run(
                [tf.assign(var, value=value) for var, value in zip(tf.trainable_variables(), saved_vars)])
            if args.print_verbosity > 1:
                print("WEIGHTS LOADED SUCCESSFULLY!")

        global dump_metadata
        if dump_metadata:
            num_trainable_weights = sum([np.prod(np.array([dim.value for dim in v.shape.dims])) for v in tf.trainable_variables()])
            make_sure_path_exists("outputs")
            with open('outputs/' + 'metadata_' + confname + '.pkl', 'wb') as f:
                pickle.dump([params, num_trainable_weights], f, pickle.HIGHEST_PROTOCOL)
            if args.print_verbosity > 0:
                pprint.pprint(params)
                print("num of rnn trainable weights: ", num_trainable_weights)
            dump_metadata = False

        early_stop_TH = 1e-3 * (M / T)
        if args.print_verbosity > 1:
            print("early_stop_TH:", early_stop_TH)
        epochs_till_converge = rnn.train(generate_batch, args.num_iters, X_validation, y_validation,
                                         auto_learning_rate_decay=True,convergence_min_delta=early_stop_TH, convergence_patience=10)

        rnn_validation_loss, _ = rnn.evaluate(X_validation, y_validation)

        if rnn_validation_loss < best_loss:
            if args.print_verbosity > 1:
                print("SAVING WEIGHTS!")
            make_sure_path_exists(weights_dir)
            with open(weights_file, 'wb') as f:
                pickle.dump(tf_session.run(tf.trainable_variables()), f, pickle.HIGHEST_PROTOCOL)

        tf.reset_default_graph()
        tf_session.close()
        runtime = time.time() - start_time

        return rnn_validation_loss, epochs_till_converge, runtime

    def best_train():
        best_loss = 1e10
        #np.random.seed(0)
        seeds = np.random.randint(0, 1000, args.rnn_num_retrainings)
        rnn_validation_loss = np.full(len(seeds), -1.0)
        runtime = np.full(len(seeds), -1.0)
        epochs_to_converge = np.full(len(seeds), -1.0)
        for j, seed in enumerate(seeds):
            rnn_validation_loss[j], epochs_to_converge[j], runtime[j] = train_models(seed, best_loss)
            if args.print_verbosity > 0:
                print("seed: %d  rnn_validation_loss: %f  num_epochs: %d runtime: %d" % (seed, rnn_validation_loss[j], epochs_to_converge[j], runtime[j]))
            best_loss = min(rnn_validation_loss[j], best_loss)
        best_j = int(np.argmin(rnn_validation_loss))
        if args.print_verbosity > 0:
            print("best_j: %d  best_seed: %d  best_loss: %f  avg_epochs: %f" % (best_j, seeds[best_j], rnn_validation_loss[best_j], np.mean(epochs_to_converge)))
        try:
            saved_vars = pickle.load(open(weights_file, 'rb'))
            tf_session = tf.Session()
            rnn = new_rnn(-1, tf_session)
            tf_session.run([tf.assign(var, value=value) for var, value in zip(tf.trainable_variables(), saved_vars)])
            if args.print_verbosity > 1:
                print("WEIGHTS LOADED SUCCESSFULLY!")
        except:
            print("ERR: failed to load weights")
        return rnn, tf_session, rnn_validation_loss[best_j], np.mean(epochs_to_converge), runtime[best_j], seeds[best_j]

    # training
    rnn, tf_session, _, avg_epochs, runtime, seed = best_train()

    # evaluation
    # common tf placeholders/tensors
    y_ = tf.placeholder(tf.int64, shape=[None, T])
    o_ = tf.placeholder(floattype, shape=[None, T, len(alphabet)])
    softmax_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=o_))
    correct_pred = tf.equal(tf.argmax(o_, axis=2), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, floattype))
    blank_accuracy = tf.reduce_mean(tf.cast(correct_pred[:, :(M + B)], floattype))
    data_accuracy = tf.reduce_mean(tf.cast(correct_pred[:, (M + B):], floattype))

    # tensorboard summaries
    loss_ph = tf.placeholder(floattype, (), "loss_ph")
    acc_ph = tf.placeholder(floattype, (), "acc_ph")
    dacc_ph = tf.placeholder(floattype, (), "dacc_ph")

    test_loss_summary = tf.summary.scalar("loss test", loss_ph)
    test_acc_summary = tf.summary.scalar("accuracy test", acc_ph)
    test_dacc_summary = tf.summary.scalar("data accuracy test", dacc_ph)
    test_summaries = tf.summary.merge([test_loss_summary, test_acc_summary, test_dacc_summary])

    # baseline predictions
    rr_loss = (M * np.log(n)) / (B+2*M)
    rr_acc = (M+B+M/n) / T
    rr_dacc = 1 / n

    # rnn evaluation
    _, rnn_test_o = rnn.predict(X_test)
    num_batches = test_size // batch_size
    cum_loss = -1.0
    cum_acc  = -1.0
    cum_dacc = -1.0
    for i in range(num_batches):
        o_batch = rnn_test_o[i * batch_size:(i + 1) * batch_size]
        y_batch = y_test[i * batch_size:(i + 1) * batch_size]
        batch_loss, batch_acc, batch_dacc = tf_session.run([softmax_cross_entropy, accuracy, data_accuracy],
                                                    feed_dict={y_: y_batch, o_: o_batch})
        cum_loss = batch_loss if cum_loss < 0 else cum_loss + batch_loss
        cum_acc = batch_acc if cum_acc < 0 else cum_acc + batch_acc
        cum_dacc = batch_dacc if cum_dacc < 0 else cum_dacc + batch_dacc
    rnn_test_loss = cum_loss / num_batches
    rnn_test_acc = cum_acc / num_batches
    rnn_test_dacc = cum_dacc / num_batches

    test_summary = tf_session.run(test_summaries, feed_dict={loss_ph: rnn_test_loss, acc_ph: rnn_test_acc, dacc_ph: rnn_test_dacc})
    tb_writer.add_summary(test_summary, B if args.write_delay_time else M)
    tb_writer.flush()

    if args.print_verbosity > 1 and T < 1000:
        fully_rand_o = np.random.random((X_test.shape[0], T, len(alphabet)))
        y_test_one_hot = to_one_hot(y_test, tf_session)
        perfect_prediction_o = np.log(y_test_one_hot + np.full(y_test_one_hot.shape, 1e-8))

        # print a few outputs of each predictor
        predictors = [('fully_rand', fully_rand_o),
                      ('perfect_prediction', perfect_prediction_o), (params['rnn_cell'], rnn_test_o)]
        num_samples_to_print = 5
        shift_str = ((max([len(name) for name, _ in predictors]) - 4 + 1) * " ")
        print("X_test: " + shift_str, idx_to_str(X_test[:num_samples_to_print]) if args.to_one_hot_on_the_fly else one_hots_to_str(X_test[:num_samples_to_print]))
        print("y_test: " + shift_str, idx_to_str(y_test[:num_samples_to_print]))
        for predictor_name, o in predictors:
            shift_str = ((max([len(name) for name, _ in predictors]) - len(predictor_name) + 1) * " ")
            print(predictor_name + "_y: " + shift_str,
                  one_hots_to_str(o[:num_samples_to_print]))

        # print loss and accuracies of all predictors
        for predictor_name, o in predictors:
            loss, acc, bacc, dacc = tf_session.run([softmax_cross_entropy, accuracy, blank_accuracy, data_accuracy],
                           feed_dict={y_: y_test, o_: o})
            shift_str = ((max([len(name) for name, _ in predictors]) - len(predictor_name) + 1) * " ")
            print(predictor_name + ": " + shift_str,
                  "  loss: %f  accuracy: %f  blank_accuracy: %f  data_accuracy: %f" % (loss, acc, bacc, dacc))
        print("rand_recaller theoretic loss:", (M * np.log(n)) / (B+2*M))

    return rr_loss, rr_acc, rr_dacc, rnn_test_loss, rnn_test_acc, rnn_test_dacc, avg_epochs, runtime, seed


if params['B'] == -1:
    # run on different B's
    #Bs = [x for x in range(0, 50, 5)] + [x for x in range(50, 101, 10)]
    Bs = [x for x in range(0, 201, 20)]
    rr_loss = np.full(len(Bs), -1.0)
    rr_acc = np.full(len(Bs), -1.0)
    rr_dacc = np.full(len(Bs), -1.0)
    rnn_test_loss = np.full(len(Bs), -1.0)
    rnn_test_acc = np.full(len(Bs), -1.0)
    rnn_test_dacc = np.full(len(Bs), -1.0)
    num_epochs = np.full(len(Bs), -1.0)
    runtime = np.full(len(Bs), -1.0)
    best_seed = np.full(len(Bs), -1.0)
    for i, b in enumerate(Bs):
        rr_loss[i], rr_acc[i], rr_dacc[i], rnn_test_loss[i], rnn_test_acc[i], rnn_test_dacc[i], num_epochs[i], runtime[i], best_seed[i] = train_and_evaluate(b)
    print("Bs: ", Bs)
    print("num_epochs", num_epochs)
    print("runtime: ", runtime)
    print("total runtime: ", sum(runtime))
    print("best_seed: ", best_seed)

    save_to = "outputs/run_index" + str(args.run_index) + ".npz"
    np.savez(save_to, Bs=Bs, rr_loss=rr_loss, rr_acc=rr_acc, rr_dacc=rr_dacc,
             rnn_test_loss=rnn_test_loss, rnn_test_acc=rnn_test_acc, rnn_test_dacc=rnn_test_dacc,
             num_epochs=num_epochs, best_seed=best_seed)

else:
    # run on a single B
    train_and_evaluate(params['B'])
