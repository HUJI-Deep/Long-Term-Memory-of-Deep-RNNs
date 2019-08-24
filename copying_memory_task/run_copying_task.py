from copying_task_utils import *
import argparse

parser = argparse.ArgumentParser(description='Script to run copy task with different rnn architectures')
parser.add_argument('-M', type=int, help='num of characters to memorize', default=30)
parser.add_argument('-n', type=int, help='alphabet size', default=32)
parser.add_argument('-B', type=int, help='delay time', required=True)
parser.add_argument('-num_iters', type=int, help='num training iterations', required=True)
parser.add_argument('-test_size', type=int, help='test set size', default=10000)
parser.add_argument('-validation_size', type=int, help='validation set size', default=5000)
parser.add_argument('-rnn_cell', type=str, help='RNN variant', default='scoRNN')
parser.add_argument('-rnn_depth', type=int, help='number of layers', required=True)
parser.add_argument('-rnn_hidden_dim', type=int, help='state size of each layer', required=True)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-optimizer', type=str, help='optimizer', default='RMSProp')
parser.add_argument('-learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('-generate_data', type=bool, help='regenerate data')
parser.add_argument('-load_weights', type=bool, help='load existing weights at the beginning of train', default=0)
parser.add_argument('-print_verbosity', type=int, help='verbosity of prints', default=2)

args = parser.parse_args()

alphabet = np.array([chr(x) for x in range(A_ASCII, A_ASCII + args.n)] + special_chars)
char_to_i = {c: i for (i, c) in enumerate(alphabet)}
floattype = tf.float32

confname = conf_name(args)
set_globals(args, alphabet, char_to_i, floattype, confname)
tb_writer = tf.summary.FileWriter('tb_general/' + confname)
dump_metadata = True


def train_and_evaluate():
    T = 2 * args.M + args.B  # input/output dimension

    # data preprocessing
    batch_size = args.batch_size
    validation_size = (args.validation_size // batch_size) * batch_size
    test_size = (args.test_size // batch_size) * batch_size
    X_test, y_test, X_validation, y_validation = get_data(test_size, validation_size)

    # training
    rnn, tf_session, num_epochs, runtime = train_models(X_validation, y_validation)

    # evaluation
    # common tf placeholders/tensors
    y_ = tf.placeholder(tf.int64, shape=[None, T])
    o_ = tf.placeholder(floattype, shape=[None, T, len(alphabet)])
    softmax_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=o_))
    correct_pred = tf.equal(tf.argmax(o_, axis=2), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, floattype))
    blank_accuracy = tf.reduce_mean(tf.cast(correct_pred[:, :(args.M + args.B)], floattype))
    data_accuracy = tf.reduce_mean(tf.cast(correct_pred[:, (args.M + args.B):], floattype))

    # tensorboard summaries
    loss_ph = tf.placeholder(floattype, (), "loss_ph")
    acc_ph = tf.placeholder(floattype, (), "acc_ph")
    dacc_ph = tf.placeholder(floattype, (), "dacc_ph")

    test_loss_summary = tf.summary.scalar("loss test", loss_ph)
    test_acc_summary = tf.summary.scalar("accuracy test", acc_ph)
    test_dacc_summary = tf.summary.scalar("data accuracy test", dacc_ph)
    test_summaries = tf.summary.merge([test_loss_summary, test_acc_summary, test_dacc_summary])

    # baseline predictions
    rr_loss = (args.M * np.log(args.n)) / (args.B+2*args.M)
    rr_acc = (args.M+args.B+args.M/args.n) / T
    rr_dacc = 1 / args.n

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
    tb_writer.add_summary(test_summary, args.B)
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
        print("baseline loss:", (args.M * np.log(args.n)) / (args.B+2*args.M))

    return rr_loss, rr_acc, rr_dacc, rnn_test_loss, rnn_test_acc, rnn_test_dacc, num_epochs, runtime


train_and_evaluate()
