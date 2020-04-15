
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


def train_and_test(epoch_cnt, mb_size, report):
    """train and test
    
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
    step_cnt = arrange(mb_size=mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_cnt):
        loss_list, acc_list = [], []

        for step in range(step_cnt):
            train_x, train_y = get_train_data(mb_size=mb_size, step=step)  # get mini batch for this step
            train_loss, train_acc = run_train(train_x, train_y)  # run train and get loss, acc
            loss_list.append(train_loss)
            acc_list.append(train_acc)

        if (report > 0) & ((epoch + 1) % report == 0):
            test_acc = run_test(test_x, test_y)
            print(f"""
            Epoch: {epoch+1}
            Train loss mean: {np.mean(loss_list)}
            Train accuracy mean: {np.mean(acc_list)}
            Test accuracy: {test_acc}
            """)


def arrange():
    pass


def get_train_data():
    pass


def get_test_data():
    pass


def run_train():
    pass


def run_test():
    pass


if __name__ == '__main__':
    data_path = '../data/abalone.csv'

    # dataset = load_data_set(path=data_path, input_cnt=10, output_cnt=1)
    weight_arr, bias_arr = init_model(rnd_mean=RND_MEAN, rnd_std=RND_STD, input_cnt=10, output_cnt=1)

    # print(dataset)
