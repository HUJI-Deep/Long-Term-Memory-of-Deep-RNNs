from seq_MNIST_utils import *
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.RNN_wrapper import *
from common.utils import *
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Script to run digit sum with different rnn architectures')
parser.add_argument('-permute', type=str2bool, help='permute pixels', required=True)
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-validation_size', type=int, help='validation set size', default=5000)
parser.add_argument('-rnn_cell', type=str, help='RNN variant', default='scoRNN')
parser.add_argument('-rnn_depth', type=int, help='number of layers', required=True)
parser.add_argument('-rnn_hidden_dim', type=int, help='state size of each layer', required=True)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-optimizer', type=str, help='optimizer', default='RMSProp')
parser.add_argument('-learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('-load_weights', type=str2bool, help='start training with existing weights', default=0)
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=2)
args = parser.parse_args()

input_dim = 1
output_dim = 10
image_len = 28*28
T = image_len


# load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# set sizes
batch_size = args.batch_size
test_size = (x_test.shape[0] // batch_size) * batch_size
validation_size = (args.validation_size // batch_size) * batch_size
full_train_size = (x_train.shape[0] // batch_size) * batch_size

# split data set into train, validation and test
X_validation = x_train[:validation_size]
y_validation = y_train[:validation_size]
X_train = x_train[validation_size:full_train_size]
y_train = y_train[validation_size:full_train_size]
X_test = x_test[:test_size]
y_test = y_test[:test_size]

# preprocess
update_globals(args, input_dim, output_dim, image_len, X_train, y_train)
X_validation, y_validation = preprocess(X_validation, y_validation)
X_train, y_train = preprocess(X_train, y_train)
X_test, y_test = preprocess(X_test, y_test)
update_globals(args, input_dim, output_dim, image_len, X_train, y_train)

conf_name = get_conf_name(args.rnn_cell, args.rnn_depth, args.rnn_hidden_dim,
                          args.batch_size, args.optimizer, args.learning_rate)

# init
tf_session = tf.Session()
tb_writer = tf.summary.FileWriter('tb_general/' + conf_name)
rnn = RNN(input_dim, output_dim, args.rnn_hidden_dim, T, args.rnn_depth, args.batch_size, tf_session,
          conf_name, args.rnn_cell, single_output=True, to_one_hot=False, learning_rate=args.learning_rate,
          optimizer_name=args.optimizer, tb_verbosity=1, print_verbosity=args.print_verbosity)

convergence_min_delta = 1e-3
if args.print_verbosity > 1:
    print("min_delta %f" % convergence_min_delta)

# train
tf_session.run(tf.global_variables_initializer())
num_iterations = rnn.train(get_batch, args.num_iters, X_validation, y_validation, load_weights=args.load_weights,
                           auto_learning_rate_decay=True, convergence_min_delta=convergence_min_delta,
                           convergence_patience=(5 if args.batch_size > 256 else 10))

# tensorboard summaries
test_loss_summary = tf.summary.scalar("loss test", rnn.loss_by_ph)
test_acc_summary = tf.summary.scalar("accuracy test", rnn.accuracy_by_ph)
test_summaries = tf.summary.merge([test_loss_summary, test_acc_summary])

validation_loss_summary = tf.summary.scalar("loss validation", rnn.loss_by_ph)
validation_acc_summary = tf.summary.scalar("accuracy validation", rnn.accuracy_by_ph)
validation_summaries = tf.summary.merge([validation_loss_summary, validation_acc_summary])

# evaluate
test_yhat, o_test = rnn.predict(X_test)
validation_yhat, o_validation = rnn.predict(X_validation)
test_loss, test_acc, test_summary = tf_session.run([rnn.loss_by_ph, rnn.accuracy_by_ph, test_summaries],
                                         feed_dict={rnn.logits_ph_: o_test, rnn.y_: y_test})
validation_loss, validation_acc, validation_summary = \
    tf_session.run([rnn.loss_by_ph, rnn.accuracy_by_ph, validation_summaries],
                                         feed_dict={rnn.logits_ph_: o_validation, rnn.y_: y_validation})
tb_writer.add_summary(test_summary, T)
tb_writer.add_summary(validation_summary, T)
tb_writer.flush()

# print results
if args.print_verbosity > 1:
    num_samples_to_print = 20
    print("y_test     " + len(args.rnn_cell)*" ", np.reshape(y_test, (-1))[:num_samples_to_print].astype(np.int32))
    print(args.rnn_cell + ": yhat_test", np.reshape(test_yhat, (-1))[:num_samples_to_print])
if args.print_verbosity > 0:
    print("test accuracy: %f" % test_acc)
    print("validation accuracy: %f" % validation_acc)

# close sessions
tf.reset_default_graph()
tf_session.close()
