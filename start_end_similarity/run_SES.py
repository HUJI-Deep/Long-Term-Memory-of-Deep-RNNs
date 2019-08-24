import argparse
import numpy.matlib
from SES_utils import *

parser = argparse.ArgumentParser(description='Script to run copy task with different rnn architectures')
parser.add_argument('-T', type=int, help='sequence length', required=True)
parser.add_argument('-n', type=int, help='alphabet size', default=32)
parser.add_argument('-m', type=int, help='num of characters to compare', default=30)
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-test_size', type=int, help='test set size', default=10000)
parser.add_argument('-validation_size', type=int, help='validation set size', default=5000)
parser.add_argument('-rnn_cell', type=str, help='RNN variant', default='scoRNN')
parser.add_argument('-rnn_depth', type=int, help='number of layers', required=True)
parser.add_argument('-rnn_hidden_dim', type=int, help='state size of each layer', required=True)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-optimizer', type=str, help='optimizer', default='RMSProp')
parser.add_argument('-learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('-load_weights', type=bool, help='start training with exisiting weights', default=0)
parser.add_argument('-generate_data', type=bool, help='regenerate data')
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=2)
parser.add_argument('-tb_verbosity', type=int, help='verbosity of tensorboard logging', default=1)

args = parser.parse_args()
C = 2 # num of classes is C+1
A_ASCII = 65
ASCII_MAX = 126
print(args)
assert args.m <= args.T, 'm must be smaller than T//2'
assert args.m >= 2, 'the minimal valid m is 2'
alphabet = np.array([chr(x) for x in range(A_ASCII, A_ASCII + args.n)] + (['-'] if (args.m < args.T//2) else []))
char_to_i = {c: i for (i, c) in enumerate(alphabet)}
conf_name = get_conf_name(args)

tb_writer = tf.summary.FileWriter('tb_general/' + conf_name)


def train_and_evaluate():
    # set utils globals
    set_globals(C, args, alphabet, char_to_i)
    # get data
    batch_size = args.batch_size
    test_size = (args.test_size // batch_size) * batch_size
    validation_size = (args.validation_size // batch_size) * batch_size
    X_test, y_test, X_validation, y_validation = get_data(test_size, validation_size)

    # train
    rnn, num_iterations, tf_session = train_rnn(X_validation, y_validation, conf_name=conf_name)

    # tensorboard summaries
    test_loss_summary = tf.summary.scalar("loss test", rnn.loss_by_ph)
    test_acc_summary = tf.summary.scalar("accuracy test", rnn.accuracy_by_ph)
    test_summaries = tf.summary.merge([test_loss_summary, test_acc_summary])

    # evaluate
    test_yhat, o_test = rnn.predict(X_test)
    test_loss, test_acc, test_summary = tf_session.run([rnn.loss_by_ph, rnn.accuracy_by_ph, test_summaries],
                                             feed_dict={rnn.logits_ph_: o_test, rnn.y_: y_test})
    tb_writer.add_summary(test_summary, args.T)
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


train_and_evaluate()
