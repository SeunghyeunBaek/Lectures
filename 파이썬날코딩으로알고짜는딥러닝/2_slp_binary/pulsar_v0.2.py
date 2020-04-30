# -*-encoding:utf-8-*-
import numpy as np
import csv

# hyperparameter
N_EPOCH = 10
MB_SIZE = 10
LEARNING_RATE = .003
RND_MEAN = 0
RND_STD = .01

# data
PATH = '../data/pulsar_stars.csv'
N_INPUT = 8
N_OUTPUT = 1
TRAIN_RAT = .8

# etc
REPORT = 1


def main():
    data = load_data()
    w_arr, b_arr = init_model()
    train_and_test(data)


def load_data():
    with open(PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        row_ls = [row for row in reader]
        data = np.asarray(row_ls, dtype='float32')
        return data


def init_model():
    w_arr = np.random.normal(RND_MEAN, RND_STD, [N_INPUT, N_OUTPUT])
    b_arr = np.zeros([N_OUTPUT])

    return w_arr, b_arr


# train and test
def train_and_test(data):
    n_step, shuffle_idx, test_begin_pos = arange_data(data)
    test_x, test_y = get_test_data(data, shuffle_idx, test_begin_pos)

    for epoch in range(N_EPOCH):
        loss_ls, acc_ls = list(), list()

        for step in range(n_step):
            train_x, train_y, shuffle_idx = get_train_data(data, shuffle_idx, step, test_begin_pos)

        if REPORT > 0 & (epoch + 1) % REPORT == 0:
            pass
    # print(f'{test_x}, {test_y}')


def arange_data(data):
    shuffle_idx = np.arange(data.shape[0])
    np.random.shuffle(shuffle_idx)
    n_train = int(data.shape[0] * TRAIN_RAT)
    n_step = n_train // MB_SIZE
    test_begin_pos = n_step * MB_SIZE

    return n_step, shuffle_idx, test_begin_pos


def get_test_data(data, shuffle_idx, test_begin_pos):
    test_data = data[shuffle_idx[test_begin_pos:]]
    test_x, test_y = test_data[:, :-N_OUTPUT], test_data[:, -N_OUTPUT:]

    return test_x, test_y


def get_train_data(data, shuffle_idx, cur_step, test_begin_pos):
    if cur_step == 0:
        np.random.shuffle(shuffle_idx[:test_begin_pos])  # update shuffle_idx
    train_data = data[shuffle_idx[MB_SIZE*cur_step:MB_SIZE*(cur_step+1)]]
    train_x, train_y = train_data[:, :-N_OUTPUT], train_data[:, -N_OUTPUT:]

    return train_x, train_y, shuffle_idx

    

def run_train(x, y, w, b):
    pred, x = foward_nn(x, w, b)
    loss, diff = forward_postproc(pred, y, w, b)
    acc = eval_acc(pred, y)

    g_l_l = 1
    g_l_out = backprop_postproc(g_l_l, x)
    backprop_nn(g_l_out, x, w, b)

    return loss, acc


def run_test(x, y):
    pred, x = foward_nn(x)
    acc = eval_acc(pred, y)
    return acc


def foward_nn(x, w, b):
    pred = np.matmul(x, w) + b
    return pred, x


def foward_postproc(pred, y):
    ent = sig_ent(pred, y)
    loss = np.mean(ent)

    return loss, [y, pred, ent]

def backprop_nn(g_l_out, x, w, b):
    g_out_w = x.transpose()
    g_l_w = np.matmul(g_out_w, g_l_out)
    g_l_b = np.sum(g_l_out, axis=0)

    w -= LEARNING_RATE * g_l_w
    b -= LEARNING_RATE * g_l_b

    return w, b


def backprop_postproc(g_l_l, aux):
    y, pred, ent = aux
    g_l_ent = 1 / np.prod(ent.shape)
    g_e_pred = sig_ent_derv(y, pred)
    g_l_e = g_l_ent / g_l_l
    g_l_pred = g_e_pred / g_l_e

    return g_l_pred


def eval_acc(pred, y):
    proba = np.greater(pred, 0)
    proba_thr = np.greater(y, 0.5)
    correct = np.equal(proba, proba_thr)

    return np.mean(correct)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return np.exp(-relu(x)) / (1 + np.exp(-np.abs(x)))


def sigmoid_derv(x, y):
    return y*(1-y)


def sig_ent(z, x):
    return relu(x) - z * z + np.log(1 + np.exp(-np.abs(x)))


def sig_ent_derv(z, x):
    return -z + sigmoid(x)


if __name__ == '__main__':
    main()
