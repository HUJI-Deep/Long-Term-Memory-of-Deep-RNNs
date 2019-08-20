import os, sys, errno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RNN_wrapper import *
from copy_task_utils import *
import argparse
import fcntl
import time
import pickle

parser = argparse.ArgumentParser(description='Script to run copy task with different rnn architectures')
parser.add_argument('-M', type=int, help='num of characters to memorize', default=30)
parser.add_argument('-n', type=int, help='alphabet size', default=32)
parser.add_argument('-B', type=int, help='delay time', required=True)
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-test_size', type=int, help='test set size', required=True)
parser.add_argument('-validation_size', type=int, help='validation set size', required=True)
parser.add_argument('-rnn_cell', type=str, help='RNN variant', default='scoRNN')
parser.add_argument('-rnn_depth', type=int, help='number of layers', required=True)
parser.add_argument('-rnn_hidden_dim', type=int, help='state size of each layer', required=True)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-optimizer', type=str, help='optimizer', default='RMSProp')
parser.add_argument('-learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('-rnn_num_retrainings', type=int, help='num retrainings of rnn', default=1)
parser.add_argument('-generate_data', type=bool, help='regenerate data')
parser.add_argument('-load_weights', type=bool, help='load existing weights at the beginning of train', default=0)
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=0)

args = parser.parse_args()

M = args.M
BLANK = '-'
START_RECALL = ':'
A_ASCII = 65
ASCII_MAX = 126
special_chars = [BLANK, START_RECALL]
n = args.n  # alphabet size
alphabet = np.array([chr(x) for x in range(A_ASCII, A_ASCII + n)] + special_chars)
char_to_i = {c: i for (i, c) in enumerate(alphabet)}
floattype = tf.float32

confname = conf_name(args)
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


def train_and_evaluate(B):
    T = 2 * M + B  # input/output dimension
    weights_dir = 'rnn_weights'
    weights_file = weights_dir + '/' + 'B' + str(B) + '_' + confname + '.pkl'

    def preprocess_data(words):
        num_samples = words.shape[0]
        assert words.shape[1] == M
        x = np.column_stack((words, np.full((num_samples, B + M), char_to_i[BLANK]))).astype(np.uint32)
        x[:, T - M] = np.full((num_samples,), char_to_i[START_RECALL])
        y = np.column_stack((np.full((num_samples, M + B), char_to_i[BLANK]), words)).astype(np.int32)
        X = x
        return X, y

    def generate_batch(num_examples):
        words = np.random.randint(n, size=(num_examples, M))
        X, y = preprocess_data(words)
        return X, y

    def get_data(test_size, validation_size):
        data_filename = 'M' + str(args.M) + '_n' + str(args.n) + '_' + 'BS' + str(args.batch_size) + '_data.npz'
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
    batch_size = args.batch_size
    validation_size = (args.validation_size // batch_size) * batch_size
    test_size = (args.test_size // batch_size) * batch_size
    X_test, y_test, X_validation, y_validation = get_data(test_size, validation_size)

    def new_rnn(seed, tf_session):
        extended_conf_name = "B" + str(B) + "_" + confname + "_seed" + str(seed)
        rnn = RNN(len(alphabet), len(alphabet), args.rnn_hidden_dim, T, args.rnn_depth, batch_size,
                  tf_session,
                  extended_conf_name, cell_name=args.rnn_cell,
                  to_one_hot=True,
                  learning_rate=args.learning_rate,
                  optimizer_name=args.optimizer,
                  tb_verbosity=1, log_period=100,
                  print_verbosity=args.print_verbosity)
        return rnn

    def run_train(rnn, tf_session):
        start_time = time.time()
        if args.load_weights:
            saved_vars = pickle.load(open(weights_file, 'rb'))
            tf_session.run(
                [tf.assign(var, value=value) for var, value in zip(tf.trainable_variables(), saved_vars)])
            if args.print_verbosity > 1:
                print("WEIGHTS LOADED SUCCESSFULLY!")

        early_stop_TH = 1e-3 * (M / T)
        if args.print_verbosity > 1:
            print("early_stop_TH:", early_stop_TH)
        epochs_till_converge = rnn.train(generate_batch, args.num_iters, X_validation, y_validation,
                                         auto_learning_rate_decay=True, convergence_min_delta=early_stop_TH, convergence_patience=10)
        runtime = time.time() - start_time

        return epochs_till_converge, runtime

    def train_models():
        seed = np.random.randint(0, 1000)
        tf_session = tf.Session()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        rnn = new_rnn(seed, tf_session)
        tf_session.run(tf.global_variables_initializer())
        epochs_to_converge, runtime = run_train(rnn, tf_session)

        if args.print_verbosity > 0:
            print("seed: %d  num_epochs: %d  runtime: %d" % (seed, epochs_to_converge, runtime))

        return rnn, tf_session, epochs_to_converge, runtime, seed

    # training
    rnn, tf_session, num_epochs, runtime, seed = train_models()

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
    tb_writer.add_summary(test_summary, B)
    tb_writer.flush()

    if args.print_verbosity > 1 and T < 1000:
        # print a few data samples
        predictors = [(args.rnn_cell, rnn_test_o)]
        num_samples_to_print = 5
        shift_str = (len(args.rnn_cell) - 4 + 1) * " "
        print("X_test: " + shift_str, idx_to_str(X_test[:num_samples_to_print]))
        print("y_test: " + shift_str, idx_to_str(y_test[:num_samples_to_print]))
        shift_str = ((len(args.rnn_cell) - len(args.rnn_cell) + 1) * " ")
        print(args.rnn_cell + "_y: " + shift_str,
              one_hots_to_str(rnn_test_o[:num_samples_to_print]))

        # print loss and accuracies
        loss, acc, bacc, dacc = tf_session.run([softmax_cross_entropy, accuracy, blank_accuracy, data_accuracy],
                       feed_dict={y_: y_test, o_: rnn_test_o})
        print(args.rnn_cell + ": " + shift_str,
              "  loss: %f  accuracy: %f  blank_accuracy: %f  data_accuracy: %f" % (loss, acc, bacc, dacc))
        print("baseline loss:", (M * np.log(n)) / (B+2*M))

    return rr_loss, rr_acc, rr_dacc, rnn_test_loss, rnn_test_acc, rnn_test_dacc, num_epochs, runtime, seed


train_and_evaluate(args.B)
