
# libraries
import numpy as np
import time
import csv

# config
np.random.seed(42)  # get random seed


# get random seed based on time
def randomize():
    np.random.seed(time.time())


# set hyperparmeters
RND_MEAN = 0
RND_STD = .003
MB_SIZE = 10
LEARNING_RATE = .001
data_path = '../data/abalone.csv'


# main
def main(epoch_cnt=10, mb_size=10, report=1):
    """main
    main function for single layered nuralnet

    Parameters
    ------------
        epoch_cnt: int
            number of epoches
        mb_size: int
            mini batch size
        report: int
            interval for reporting accuracy

    Returns
    --------
        None
    """
    dataset = load_data_set(path=data_path, input_cnt=10, output_cnt=1)
    weight_arr, bias_arr = init_model(rnd_mean=RND_MEAN, rnd_std=RND_STD, input_cnt=10, output_cnt=1)
    train_and_test()

    return None


def load_data_set(path, input_cnt, output_cnt):
    """load data set
    load data set from path
    make one hot vector with gender column

    Parameters
    -----------
    input_cnt: int
        number of x columns
    output_cnt: int
        number of target columns

    Returns
    --------
    data: numpy ndarray
    """
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)  # skip header
        row_list = []
        for n, row in enumerate(csv_reader):
            row_list.append(row)
    # initiate data array
    data = np.zeros(shape=[len(row_list), input_cnt+output_cnt])

    for n, row in enumerate(row_list):
        # make one hot vector for gender column
        if row[0] == 'I':
            data[n, 0] = 1
        elif row[0] == 'M':
            data[n, 1] = 1
        elif row[0] == 'F':
            data[n, 2] = 1
        else:
            print("invalid gender")

    return data


def init_model(rnd_mean, rnd_std, input_cnt, output_cnt):
    """initiate model parameters
        initiate weight, bias with hyperparameters

    Parameters
    ------------
        rnd_mean: float
        rnd_std: float
        input_cnt: int
        output_cnt: int

    Returns
    --------
        weigh_arr: np.ndarray
        bias_arr: np.adarray
    """
    weight_arr = np.random.normal(loc=rnd_mean, scale=rnd_std, size=[input_cnt, output_cnt])
    bias_arr = np.zeros(shape=[output_cnt])

    return weight_arr, bias_arr


def train_and_test(data, epoch_cnt, mb_size, report):
    """train and test
    run nuralnet

    Parameters
    -----------
    epoch_cnt: int
        number of epoches
    mb_size: int
        mini batch size
    report: int
        print test loss, accuracy each `report` count

    Returns
    --------
    """
    shuffle_map, step_cnt, test_begin_idx = arange_data(mb_size=mb_size)
    test_x, test_y = get_test_data(data=data,
                                      shuffle_map=shuffle_map,
                                      test_begin_idx=test_begin_idx,
                                      output_cnt=1)

    for epoch in range(epoch_cnt):
        loss_list, acc_list = [], []
        # get mini batch for this step
        for step in range(step_cnt):
            train_x, train_y = get_train_data(data=data,
                                                 mb_size=mb_size,
                                                 shuffle_map=shuffle_map,
                                                 test_begin_idx=test_begin_idx,
                                                 output_cnt=1,
                                                 step_idx=step)
            # run train and get loss, acc
            train_loss, train_acc = run_train(train_x, train_y)
            loss_list.append(train_loss)
            acc_list.append(train_acc)

        if (report > 0) & ((epoch + 1) % report == 0):
            test_acc = run_test(test_x, test_y)
            print(f"""
            Epoch: {epoch+1}
            Train loss mean: {np.mean(loss_list)}
            Train accuracy mean: {np.mean(acc_list)}
            Test accuracy: {test_acc}
            =============================================
            /n
            """)

    final_acc = run_test(test_x, test_y)
    print(f"""
    Final Test
    Accuracy = {final_acc}
    """)


def arange_data(data, mb_size):
    """shuffle data

    Parameters
    -----------
    mb_size: int
        mini batch size

    Returns
    --------
    shuffle_map: numpy ndarray
    step_count: int
        number of batch steps for each epoch
    test_begin_idx: int
    """

    shuffle_map = np.arange(data.shape[0])  # get row index
    np.random.shuffle(shuffle_map)  # shuffle row index
    step_cnt = int(data.shape[0] * .8 // mb_size)  # get step count(train 80%)
    test_begin_idx = step_cnt * mb_size

    return shuffle_map, step_cnt, test_begin_idx


def get_train_data(data, mb_size, shuffle_map, test_begin_idx, output_cnt, step_idx):
    """get train data
    get mini batch data for the batch step

    Parameters
    -----------
    data: numpy ndarray
    mb_size: int
    shuffle_mape: numpy ndarray
    test_begin_idx: int
    output_cnt: int
    step_idx: int
    """
    if step_idx == 0:
        # shuffle and remove test index for each epoch
        np.random.shuffle(shuffle_map[:test_begin_idx])
    # get batch train data
    train_data = data[shuffle_map[mb_size * step_idx:mb_size * (step_idx + 1)]]
    train_x, train_y = train_data[: -output_cnt], train_data[:, -output_cnt:]
    return train_x, train_y


def get_test_data(data, shuffle_map, test_begin_idx, output_cnt):
    """get test data

    Parameters
    -----------
    data: numpy ndarray
    shuffle_map: numpy array
        shuffled index of the data
    test_begin_idx: int
    output_cnt: int

    Returns
    --------
    test_x: numpy ndarray
    test_y: numpy ndarray
    """

    test_data = data[shuffle_map[test_begin_idx:]]
    test_x, test_y = test_data[:, :-output_cnt], test_data[:, -output_cnt:]
    return test_x, test_y


def run_train(x, y):
    # foward
    out_y, x = foward_nn(x=x)  # get out_y and return out_y and x
    loss, diff = forward_postproc(out_y=out_y, y=y)  # get loss
    accuracy = eval_accuracy(out_y=out_y, y=y)

    # back
    loss_grd = 1  # dL/dL
    out_grd = backprop_postproc(g_loss=g_loss, aux_pp=aux_pp)  # dL/dx
    backprop_nn(g_out, aux_nn)

    return loss, accuracy


def foward_nn(x, weight_arr, bias_arr):
    """foward neuralnet

    Parameters
    -----------
    x: numpy ndarray
        n_rows: number of batches
        n_cols: number of features
    weight_arr: numpy ndarray
        n_rows: number of features
        n_cols: number of layers
    bias_arr: numpy adarray

    Returns
    --------
    out_y: numpy ndarry
        n_rows: number of features
        n_cols: number of batches

    """
    out_y = np.matmul(x, weight_arr) + bias_arr
    return out_y, x


def forward_postproc(out_y, y):
    """foward neuralnet post process

    Parameters
    ------------
    out_y: numpy ndarray
    y: numpy ndarray

    Returns
    --------
    loss: float
    diff: float
    """
    diff = out_y - y
    square = np.square(diff)
    loss = np.mean(square)

    return loss, diff


def backprop_nn(dl_dout_arr, weight_arr, bias_arr, x):
    """back propagatagtion

    Parameters
    ------------
    dl_dout_arr: numpy ndarray
        순전파 출력에 대한 Loss 기울기 (dl/dout)
        r_row : number of batches
        n_col: number of output

    weight_arr: numpy ndarray
        n_row: number of features
        n_col: number of output

    bias_arr: numpy ndarray
        n_row: number of batches
        n_col: number of output

    Returns
    --------
    weight_arr: numpy ndarray
    bias_arr: numpy ndarray
    """

    dout_dw_arr = x.transpose()  # dout/dw
    dl_dw_arr = np.matmul(dout_dw_arr, dl_dout_arr)  # dl/dout

    dl_db_arr = np.sum(dl_dout_arr)  # dl/db

    # update weight, bias with learning rate
    weight_arr = LEARNING_RATE * dl_dw_arr
    bias_arr = LEARNING_RATE * dl_db_arr

    return weight_arr, bias_arr


def run_test():
    pass


if __name__ == '__main__':
    data_path = '../data/abalone.csv'

    # dataset = load_data_set(path=data_path, input_cnt=10, output_cnt=1)
    # weight_arr, bias_arr = init_model(rnd_mean=RND_MEAN, rnd_std=RND_STD, input_cnt=10, output_cnt=1)
    # shuffle_map, step_cnt, test_begin_idx = arange_data(data=dataset, mb_size=MB_SIZE)
    # test_x, test_y = get_test_data(data=dataset,
    #                                   shuffle_map=shuffle_map,
    #                                   test_begin_idx=test_begin_idx,
    #                                   output_cnt=1)
    # train_x, train_y = get_train_data(data=dataset,
    #                                          mb_size=MB_SIZE,
    #                                          shuffle_map=shuffle_map,
    #                                          test_begin_idx=test_begin_idx,
    #                                          output_cnt=1,
    #                                          step_idx=5)

    # 