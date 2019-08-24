import numpy as np
from sklearn.utils import shuffle


xpermutation = np.random.RandomState(92916).permutation(784)
args = None
input_dim = -1
output_dim = -1
image_len = -1
batch_idx = 0
num_batches = -1
X_train = None
y_train = None


def get_conf_name(cell, depth, width, bs, optimizer, eta):
    return cell + "_depth" + str(depth) + "_width" + str(width) + "_BS" + str(bs) + "_" + optimizer + "_eta%.4f" % eta


def update_globals(args_, input_dim_, output_dim_, image_len_, X_train_, y_train_):
    global args, input_dim, output_dim, image_len, num_batches, X_train, y_train
    args = args_
    input_dim = input_dim_
    output_dim = output_dim_
    image_len = image_len_
    X_train = X_train_
    y_train = y_train_

    num_batches = X_train.shape[0] // args.batch_size


def preprocess(X, y):
    X_expanded = np.reshape(X, (-1, image_len, input_dim))
    X_continuous = X_expanded / 255.0
    if args.permute:
        X_continuous_permuted = X_continuous[:, xpermutation]
    y_expanded = np.expand_dims(y, axis=1)
    return (X_continuous_permuted if args.permute else X_continuous), y_expanded


def get_batch(bs):
    global batch_idx, X_train, y_train
    X = X_train[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
    y = y_train[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
    batch_idx = (batch_idx+1) % num_batches
    if batch_idx == 0:
        X_train, y_train = shuffle(X_train, y_train, random_state=np.random.randint(0, 1000))
    return X, y
