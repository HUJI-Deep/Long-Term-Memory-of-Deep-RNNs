from seq_MNIST_conf_file import *
from seq_MNIST_utils import *
from sklearn.utils import shuffle
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Script to run digit sum with different rnn architectures')
parser.add_argument('-permute', type=bool, help='permute pixels', default=False)
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-conf_index', type=int, help='index of architecture conf', required=True)
parser.add_argument('-validation_size', type=int, help='validation set size', default=5000)
parser.add_argument('-rnn_num_retrainings', type=int, help='num retrainings of the same configuration', default=1)
parser.add_argument('-load_weights', type=bool, help='start training with existing weights', default=0)
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=0)
args = parser.parse_args()

params = get_conf(args.conf_index)
print(params)

# input_dim = 256
# input_bits = 8
input_dim = 1
output_dim = 10
image_len = 28*28
T = image_len

if args.permute:
    permute = np.random.RandomState(92916)
    xpermutation = permute.permutation(784)


def preprocess(X, y):
    X_expanded = np.reshape(X, (-1, image_len, input_dim))
    X_continuous = X_expanded / 255.0
    if args.permute:
        X_continuous_permuted = X_continuous[:, xpermutation]
    y_expanded = np.expand_dims(y, axis=1)
    return (X_continuous_permuted if args.permute else X_continuous), y_expanded

# load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# set sizes
batch_size = params['rnn_batch_size']
test_size = (x_test.shape[0] // batch_size) * batch_size
validation_size = (args.validation_size // batch_size) * batch_size
full_train_size = (x_train.shape[0] // batch_size) * batch_size

X_validation = x_train[:validation_size]
y_validation = y_train[:validation_size]
X_train = x_train[validation_size:full_train_size]
y_train = y_train[validation_size:full_train_size]
X_test = x_test[:test_size]
y_test = y_test[:test_size]

# preprocess
X_validation, y_validation = preprocess(X_validation, y_validation)
X_train, y_train = preprocess(X_train, y_train)
X_test, y_test = preprocess(X_test, y_test)

conf_name = get_conf_name(params['rnn_cell'], params['rnn_depth'], params['rnn_hidden_dim'],
                          params['rnn_batch_size'], params['optimizer'], params['rnn_learning_rate'],
                          params['rnn_hidden_eta'], params['rms_decay'])

batch_idx = 0
num_batches = X_train.shape[0] // batch_size


def get_batch(bs):
    global batch_idx, X_train, y_train
    X = X_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    y = y_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    batch_idx = (batch_idx+1) % num_batches
    if batch_idx == 0:
        X_train, y_train = shuffle(X_train, y_train, random_state=np.random.randint(0, 1000))
    return X, y

# init
tf_session = tf.Session()
tb_writer = tf.summary.FileWriter('tb_general/' + conf_name)
rnn = RNN(input_dim, output_dim, params['rnn_hidden_dim'], T, params['rnn_depth'], params['rnn_batch_size'], tf_session,
                  "T" + str(T) + "_" + conf_name,
                  params['rnn_cell'], single_output=True, logspace=False, to_one_hot=False,
                  learning_rate=params['rnn_learning_rate'],
                  hidden_eta=params['rnn_hidden_eta'],
                  optimizer_name=params['optimizer'], rms_decay=params['rms_decay'],
                  tb_verbosity=1, print_verbosity=args.print_verbosity)

convergence_min_delta = 1e-3
if args.print_verbosity > 1:
    print("min_delta %f" % convergence_min_delta)

# train
tf_session.run(tf.global_variables_initializer())
if args.load_weights:
    rnn.load_weights()
else:
    num_iterations = rnn.train(get_batch, args.num_iters, X_validation, y_validation, auto_learning_rate_decay=True,
           convergence_min_delta=convergence_min_delta, convergence_patience=(5 if params['rnn_batch_size'] > 256 else 10))

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
validation_loss, validation_acc, validation_summary = tf_session.run([rnn.loss_by_ph, rnn.accuracy_by_ph, validation_summaries],
                                         feed_dict={rnn.logits_ph_: o_validation, rnn.y_: y_validation})
tb_writer.add_summary(test_summary, T)
tb_writer.add_summary(validation_summary, T)
tb_writer.flush()

if args.print_verbosity > 1:
    num_samples_to_print = 10
    print("X_test", X_test[:num_samples_to_print])
    print("y_test   ", np.reshape(y_test, (-1))[:num_samples_to_print].astype(np.int32))
    print("yhat_test", np.reshape(test_yhat, (-1))[:num_samples_to_print])
if args.print_verbosity > 0:
    print("test accuracy: %f" % test_acc)
    print("validation accuracy: %f" % validation_acc)

# close sessions
tf.reset_default_graph()
tf_session.close()