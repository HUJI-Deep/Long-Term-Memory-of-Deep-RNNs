from common.scoRNN import *
from common.utils import *

name_to_cell = {'BasicRNN': tf.contrib.rnn.BasicRNNCell, 'LSTM': tf.nn.rnn_cell.LSTMCell, 'scoRNN': scoRNNCell}
name_to_optimizer = {'ADAM': tf.train.AdamOptimizer, 'SGD': tf.train.GradientDescentOptimizer, 'RMSProp': tf.train.RMSPropOptimizer}


def bias_variable(shape, initializer=None, name=None):
    initial = initializer if initializer is not None else tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


class RNN:
    def __init__(self, input_dim, output_dim, hidden_dim, T, depth, batch_size,
                 tf_session, configuration_name, cell_name, single_output=False, regression_loss=False,
                 to_one_hot=False, learning_rate=1e-3, hidden_eta=1e-4,
                 optimizer_name='RMSProp', rms_decay=0.9, grad_clipping_TH=10.0,
                 tb_verbosity=1, log_period=100, print_verbosity=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.depth = depth
        self.batch_size = batch_size
        self.tf_session = tf_session
        self.cell_name = cell_name
        layers = []
        for l in range(1, self.depth + 1):
            with tf.variable_scope("layer%d" % l):
                if cell_name == 'scoRNN':
                    self.D = np.diag(np.concatenate([np.ones(self.hidden_dim//2), \
                                            -np.ones(self.hidden_dim - self.hidden_dim//2)]))
                    layers.append(name_to_cell[self.cell_name](self.hidden_dim, D=self.D))
                else:
                    layers.append(name_to_cell[self.cell_name](self.hidden_dim))
        self.cell = tf.contrib.rnn.MultiRNNCell(layers)
        self.single_output = single_output
        self.regression_loss = regression_loss
        if self.regression_loss:
            assert self.output_dim == 1, "regression loss is currently implemented only for scalars"
        self.num_outputs = 1 if self.single_output else self.T
        self.floattype = tf.float64 if cell_name == 'OrthogonalRNN' else tf.float32
        if self.cell_name == 'RAC':
            self.init_state = tuple([tf.fill([self.batch_size, self.hidden_dim], float('NaN')) for _ in
                                 range(self.depth)])
        if self.cell_name in ['OrthogonalRNN', 'scoRNN']:
            bucket = np.sqrt(3. / self.hidden_dim)
            self.init_state = tuple([tf.random.uniform(shape=[self.batch_size, self.hidden_dim],
                                            minval=-bucket, maxval=bucket, dtype=self.floattype) for _ in range(self.depth)])
        else:
            self.init_state = self.cell.zero_state(self.batch_size, self.floattype)
        self.to_one_hot = to_one_hot
        if self.to_one_hot:
            self.X_batch_ = tf.placeholder(tf.uint8, [None, self.T], name="X_batch_")
            self.Xbatch = tf.one_hot(self.X_batch_, self.input_dim, dtype=self.floattype)
        else:
            self.X_batch_ = tf.placeholder(self.floattype, [None, self.T, self.input_dim], name="X_batch_")
            self.Xbatch = self.X_batch_
        self.y_ = tf.placeholder(self.floattype if self.regression_loss else tf.int64, shape=[None, self.num_outputs], name="y_")
        self.logits, state_outputs = self._build_rnn_graph()
        self.logits_ph_ = tf.placeholder(self.floattype, [None, self.num_outputs, self.output_dim], name="logits_ph_")
        self.predictions = tf.argmax(self.logits, axis=2)
        if self.output_dim == 1 and self.regression_loss:
            # l2 loss
            y = tf.expand_dims(self.y_, axis=-1)
            diff = (y - self.logits)
            self.loss = tf.reduce_mean(diff * diff)
            diff2 = (y - self.logits_ph_)
            self.loss_by_ph = tf.reduce_mean(diff2 * diff2)
        else:
            # cross entropy
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
            self.loss_by_ph = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits_ph_))

        self.global_step = tf.Variable(0, trainable=False)
        self.inc_global_step = tf.assign(self.global_step, self.global_step + 1)
        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=self.floattype)
        self.decrease_learning_rate = tf.assign(self.learning_rate, 0.1 * self.learning_rate)
        if optimizer_name == 'RMSProp':
            self.optimizer = (name_to_optimizer[optimizer_name])(learning_rate=self.learning_rate, decay=rms_decay)
        else:
            self.optimizer = (name_to_optimizer[optimizer_name])(learning_rate=self.learning_rate)

        if self.cell_name == 'scoRNN':
            self.hidden_eta = tf.Variable(hidden_eta, trainable=False, dtype=self.floattype)
            self.decrease_hidden_eta = tf.assign(self.hidden_eta, 0.1 * self.hidden_eta)
            self.hidden_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.hidden_eta)

            self.Wvars = [self.cell._cells[i]._W for i in range(self.depth)]
            self.Avars = [self.cell._cells[i]._A for i in range(self.depth)]
            non_hidden_vars = [v for v in tf.trainable_variables() if v not in \
                               [self.Wvars, self.Avars]]

            # Getting gradients
            self.hidden_grads_clipped = []
            self.linf_norms = []
            for grad in tf.gradients(self.loss, self.Wvars):
                # if grad has an inf val: grad = 0
                is_finite = tf.reduce_all(tf.is_finite(grad))
                g = tf.cond(is_finite, lambda: grad, lambda: tf.zeros(tf.shape(grad)))

                # grad clipping by linf norm
                self.linf_norms.append(tf.reduce_max(tf.abs(g)))
                large_norm = tf.greater(self.linf_norms[-1], grad_clipping_TH)
                self.hidden_grads_clipped.append(tf.cond(large_norm, lambda: (g / self.linf_norms[-1]) * grad_clipping_TH, lambda: g))

            # Updating variables
            self.newWs = [tf.placeholder(self.floattype, w.get_shape()) for w in self.Wvars]
            self.updateW = [tf.assign(self.Wvars[i], self.newWs[i]) for i in range(self.depth)]

            # Applying hidden-to-hidden gradients
            self.gradA = [tf.placeholder(self.floattype, a.get_shape()) for a in self.Avars]
            self.applygradA = [self.hidden_optimizer.apply_gradients([(self.gradA[l], self.Avars[l])]) for l in range(self.depth)]

            self.train_step = self.optimizer.minimize(self.loss, var_list=non_hidden_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)
        if not self.regression_loss:
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=2), self.y_), self.floattype))
            self.accuracy_by_ph = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits_ph_, axis=2), self.y_), self.floattype))
        self.tb_verbosity = tb_verbosity
        self.print_verbosity = print_verbosity

        # tensorboard
        if tb_verbosity > 0:
            if not self.regression_loss:
                accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
            loss_summary = tf.summary.scalar("loss", self.loss)
            summaries = [loss_summary] + ([accuracy_summary] if not regression_loss else [])

            self.tf_summaries = tf.summary.merge(summaries)
            self.log_period = log_period
            self.tb_writer = tf.summary.FileWriter('tb_rnn_training/' + configuration_name)

        # weights
        self.saver = tf.train.Saver()
        self.weights_dir = 'rnn_weights'
        self.weights_file = self.weights_dir + '/' + configuration_name + '.ckpt'

    def save_weights(self):
        if self.print_verbosity > 1:
            print("saving weights")
        make_sure_path_exists(self.weights_dir)
        self.saver.save(self.tf_session, self.weights_file)

    def load_weights(self):
        if self.print_verbosity > 1:
            print("loading weights")
        self.saver.restore(self.tf_session, self.weights_file)

    def _build_rnn_graph(self):
        # static RNN
        inputs = tf.unstack(self.Xbatch, num=self.T, axis=1)
        state_outputs, self.state = tf.nn.static_rnn(self.cell, inputs, initial_state=self.init_state)
        cell_o = tf.stack(state_outputs, axis=1)
        if self.single_output:
            cell_o = cell_o[:, -1, :]

        self.readout_w = tf.get_variable("readout_W", shape=[self.hidden_dim, self.output_dim],
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=self.floattype)
        self.readout_b = bias_variable([self.output_dim], initializer=tf.zeros([self.output_dim], dtype=self.floattype), name="readout_b")
        logits = tf.reshape(
            tf.matmul(tf.reshape(cell_o, [-1, self.hidden_dim]), self.readout_w)
            + self.readout_b,
            [self.batch_size, self.num_outputs, self.output_dim])
        return logits, state_outputs

    def stop_condition(self, X, y, early_stop_min_delta, early_stop_patience):
        epoch_loss, _ = self.evaluate(X, y)
        if self.tb_verbosity > 0:
            summary, global_step = self.tf_session.run([self.validation_summary, self.global_step], feed_dict={self.validation_loss_ph: epoch_loss})
            self.tb_writer.add_summary(summary, global_step)
            self.tb_writer.flush()
        if self.print_verbosity > 1:
            print("best_loss: %f  epoch_loss: %f" % (self.best_loss, epoch_loss))
        if self.best_loss - epoch_loss > early_stop_min_delta:
            self.patience_cnt = 0
            self.best_loss = epoch_loss
            self.save_weights()
        else:
            self.patience_cnt += 1

        if self.patience_cnt > early_stop_patience:
            return True
        return False

    def train(self, get_batch_func, num_iters, X_validation, y_validation, load_weights=False, auto_learning_rate_decay=False,
              early_stopping=False, check_convergence_period=1000, convergence_min_delta=1e-3, convergence_patience=5):

        if load_weights:
            self.load_weights()

        if self.tb_verbosity > 0:
            self.validation_loss_ph = tf.placeholder(self.floattype, (), "validation_loss_ph")
            self.validation_summary = tf.summary.scalar("validation_loss", self.validation_loss_ph)

        if early_stopping or auto_learning_rate_decay:
            self.best_loss = np.inf
            self.best_acc = 0.0
            self.patience_cnt = 0

        if auto_learning_rate_decay:
            self.second_phase = False

        for j in range(num_iters):
            if j % check_convergence_period == 0 or j == num_iters-1:
                if (early_stopping or auto_learning_rate_decay) and self.print_verbosity > 1:
                    print("iter: %d  patience_cnt: %d" % (j, self.patience_cnt))
                if early_stopping or auto_learning_rate_decay:
                    converged = self.stop_condition(X_validation, y_validation,
                                                      convergence_min_delta, convergence_patience)
                if early_stopping and converged:
                    if self.print_verbosity > 0:
                        print("early stopping...")
                    break

                if auto_learning_rate_decay and converged:
                    if self.second_phase:
                        print("second phase converged! early stopping...")
                        print("global_step: %d" % global_step)
                        break

                    if self.print_verbosity > 0:
                        print("converged! decreasing learning rate...")
                    global_step, old_eta = self.tf_session.run([self.global_step, self.learning_rate])

                    new_eta = self.tf_session.run(self.decrease_learning_rate)
                    if self.cell_name == 'scoRNN':
                        _ = self.tf_session.run(self.decrease_hidden_eta)
                    self.patience_cnt = 0
                    self.second_phase = True

                    if self.print_verbosity > 0:
                        print("global_step: %d  old_eta: %f  new_eta: %f" % (global_step, old_eta, new_eta))

            X_batch, y_batch = get_batch_func(self.batch_size)
            tensors = [self.train_step, self.loss, self.tf_summaries, self.global_step,
                       self.learning_rate] + ([self.accuracy] if not self.regression_loss else [])
            feed_dict = {self.X_batch_: X_batch, self.y_: y_batch}
            if self.regression_loss:
                _, loss, summary, global_step, eta = self.tf_session.run(tensors, feed_dict=feed_dict)
            else:
                if self.cell_name == 'scoRNN':
                    _, loss, summary, global_step, eta, accuracy, hidden_grads, Ws, As = \
                        self.tf_session.run(tensors + [self.hidden_grads_clipped, self.Wvars, self.Avars], feed_dict=feed_dict)
                    for l in range(self.depth):
                        DFA = Cayley_Transform_Deriv(hidden_grads[l], As[l], Ws[l], self.D)
                        self.tf_session.run(self.applygradA[l], feed_dict={self.gradA[l]: DFA})
                        A = self.tf_session.run(self.Avars[l])
                        W = makeW(A, self.D)
                        self.tf_session.run(self.updateW[l], feed_dict={self.newWs[l]: W})
                else:
                    _, loss, summary, global_step, eta, accuracy = self.tf_session.run(tensors, feed_dict=feed_dict)

            if global_step % self.log_period == 0:
                if self.print_verbosity > 0:
                    print(self.cell_name + ": global_step: %d  iteration %d:  loss: %f" % (
                        global_step, j, loss), " accuracy:", None if self.regression_loss else accuracy)
                    if self.print_verbosity > 2:
                        trainable_vars = tf.trainable_variables()
                        trainable_vars_vals = self.tf_session.run(trainable_vars)
                        for l in range(len(trainable_vars)):
                            print(trainable_vars[l].name + " (min: %f, max: %f)"
                                  % (np.min(trainable_vars_vals[l]), np.max(trainable_vars_vals[l])))
                if self.tb_verbosity > 0:
                    self.tb_writer.add_summary(summary, global_step)
                    self.tb_writer.flush()
            self.tf_session.run(self.inc_global_step)

        if early_stopping or auto_learning_rate_decay:
            self.load_weights()
        return num_iters

    def predict(self, X):
        num_samples = X.shape[0]
        assert num_samples % self.batch_size == 0, "RNN.predict() currently supports only multiples of batch_size"
        predicted_y = np.full((num_samples, self.num_outputs), -1)
        o = np.full((num_samples, self.num_outputs, self.output_dim), -1.0)
        num_batches = num_samples // self.batch_size
        for i in range(num_batches):
            X_batch = X[i*self.batch_size:(i+1)*self.batch_size]
            predicted_y[i*self.batch_size:(i+1)*self.batch_size], o[i*self.batch_size:(i+1)*self.batch_size] = \
                self.tf_session.run([self.predictions, self.logits], feed_dict={self.X_batch_: X_batch})
        return predicted_y, o

    def evaluate(self, X, y):
        num_samples = X.shape[0]
        assert num_samples % self.batch_size == 0, "RNN.evaluate() currently supports only multiples of batch_size"
        cum_acc = -1.0
        cum_loss = -1.0
        num_batches = num_samples // self.batch_size
        for i in range(num_batches):
            X_batch = X[i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
            if self.regression_loss:
                batch_loss = self.tf_session.run(self.loss, feed_dict={self.X_batch_: X_batch, self.y_: y_batch})
                batch_acc = 0
            else:
                batch_loss, batch_acc = self.tf_session.run([self.loss, self.accuracy], feed_dict={self.X_batch_: X_batch, self.y_: y_batch})
            cum_loss = batch_loss if cum_loss < 0 else cum_loss + batch_loss
            cum_acc = batch_acc if cum_acc < 0 else cum_acc + batch_acc
        loss = cum_loss / float(num_batches)
        acc = cum_acc / float(num_batches)
        return loss, acc
