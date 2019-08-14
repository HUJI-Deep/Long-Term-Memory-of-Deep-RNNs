import argparse
from start_end_similarity_conf_file import *
from start_end_similarity_utils import *

parser = argparse.ArgumentParser(description='Script to run copy task with different rnn architectures')
# parser.add_argument('-C', type=int, help='num of classes is C+1', required=True)
parser.add_argument('-m', type=int, help='num of characters to compare, T//2 if -m is set to -1', default=-1)
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-test_size', type=int, help='test set size', required=True)
parser.add_argument('-validation_size', type=int, help='validation set size', required=True)
parser.add_argument('-conf_index', type=int, help='index of architecture conf', required=True)
parser.add_argument('-rnn_num_retrainings', type=int, help='num retrainings of the same configuration', default=1)
parser.add_argument('-load_weights', type=bool, help='start training with exisiting weights', default=0)
parser.add_argument('-generate_data', type=bool, help='regenerate data')
parser.add_argument('-trigger_char', type=bool, help='add trigger char to middle of strings', default=0)
parser.add_argument('-random_position', type=int, help='place word in a random position', default=0)
parser.add_argument('-fully_rand_pos', type=int, help='spread characters uniformly at random in each half', default=0)
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=0)
parser.add_argument('-tb_verbosity', type=int, help='verbosity of tensorboard logging', default=1)
parser.add_argument('-unit_testing', type=int, help='run unit-test', default=0)


args = parser.parse_args()
A_ASCII = 65
ASCII_MAX = 126
params = get_conf(args.conf_index)
print(params)
n = params['n']  # alphabet size
m = params['T'] // 2 if args.m < 0 else args.m
assert m <= params['T']//2, 'm must be smaller than T//2'
assert m >= 2, 'the minimal valid m is 2'
# assert n <= (ASCII_MAX - A_ASCII + 1)
alphabet = np.array([chr(x) for x in range(A_ASCII, A_ASCII + n)] + (['-'] if (args.trigger_char or m < params['T']//2) else []))
char_to_i = {c: i for (i, c) in enumerate(alphabet)}
# np.random.seed(0)
conf_name = get_conf_name(params['C'], n, m, params['rnn_cell'], params['rnn_depth'], params['rnn_hidden_dim'],
                          params['rnn_batch_size'], params['optimizer'], params['rnn_learning_rate'],
                          params['rnn_hidden_eta'], params['rms_decay'])

metadata_filename = "outputs/conf_index" + str(args.conf_index) + '_metadata' + '.pkl'
tb_writer = tf.summary.FileWriter('tb_general/' + conf_name)


def train_and_evaluate():
    # T = 2 * m + params['B']  # seq len

    # set utils globals
    set_globals(n, params['C'], params['T'], m, alphabet, char_to_i,
                args.trigger_char, args.random_position, args.fully_rand_pos, args.num_iters, args.rnn_num_retrainings,
                    params, args.load_weights, args.generate_data, args.print_verbosity, args.tb_verbosity,
                    args.unit_testing, metadata_filename, params['seed'])
    # get data
    batch_size = params['rnn_batch_size']
    test_size = (args.test_size // batch_size) * batch_size
    validation_size = (args.validation_size // batch_size) * batch_size
    X_test, y_test, X_validation, y_validation = get_data(test_size, validation_size)

    # train
    rnn, _, num_epochs, tf_session = train_rnn(X_validation, y_validation, conf_name=conf_name)

    # tensorboard summaries
    test_loss_summary = tf.summary.scalar("loss test", rnn.loss_by_ph)
    test_acc_summary = tf.summary.scalar("accuracy test", rnn.accuracy_by_ph)
    test_summaries = tf.summary.merge([test_loss_summary, test_acc_summary])

    # evaluate
    test_yhat, o_test = rnn.predict(X_test)
    test_loss, test_acc, test_summary = tf_session.run([rnn.loss_by_ph, rnn.accuracy_by_ph, test_summaries],
                                             feed_dict={rnn.logits_ph_: o_test, rnn.y_: y_test})
    tb_writer.add_summary(test_summary, params['T'])
    tb_writer.flush()

    if args.print_verbosity > 1:
        num_samples_to_print = 10
        print("X_test", idx_to_str(X_test[:num_samples_to_print]))
        print("y_test   ", np.reshape(y_test, (-1))[:num_samples_to_print].astype(np.int32))
        print("yhat_test", np.reshape(test_yhat, (-1))[:num_samples_to_print])
    if args.print_verbosity > 0:
        print("test loss:     %f" % test_loss)
        print("test accuracy: %f" % test_acc)

    # close sessions
    tf.reset_default_graph()
    tf_session.close()
    return test_loss, test_acc, num_epochs


train_and_evaluate()
