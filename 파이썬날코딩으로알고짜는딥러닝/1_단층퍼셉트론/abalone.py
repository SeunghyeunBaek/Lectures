import numpy as np
import csv
import time


np.random.seed(42)  # set reandom seed


def randomize():
    np.random.seed(time.time())  # set random seed with time


# set hyperparameters
RND_MEAN = 0  # for initiating hyperparamerter
RND_STD = .003  # for initiating hyperparamerter
LEARNING_RATE = .001


# 1. main
def abalone_exec(epoch_cnt=10, mb_size=10, report=1):
    """main

    """
    load_abalone_dataset()  # 데이터 불러오기
    init_model()  # 파라메터 초가화
    train_and_test(epoch_cnt, mb_size, report)  # 모델링
    
# 2. load data
def load_abalone_dataset():
    path = '../data/abalone.csv'
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)  # 메모리에 파일 올리기
        next(csvreader, None)  # 첫행(header) 읽지 않고 건너뛰기
        rows = []
        for row in csvreader:
            rows.append(row)  # 모든 row 저장
        
        global data, input_cnt, output_cnt  # 입출력 벡터, 입력벡터 크기(열 수), 출력벡터 크기 전역변수 선언
        
        input_cnt, output_cnt = 10, 1  # 입력, 출력 벡터 수 초기화
        data = np.zeros([len(rows), input_cnt+output_cnt])  # 입출력 벡터 초기화
        
        # 데이터 전역변수에 저장
        for n, row in enumerate(rows):
            # one hot vector 
            if row[0] == 'I': data[n, 0] = 1
            if row[0] == 'M': data[n, 1] = 1
            if row[0] == 'F': data[n, 2] = 1
            data[n, 3:] = row[1: ]  # 성별정보를 제외하고 data 에 저장
                
# 파라메터 초기화

def init_model():
    global weight, bias, input_cnt, output_cnt  # 가중치, 편향, 입력벡터 차원, 출력벡터 차원 전역변수 선언
    weight = np.random.normal(loc=RND_MEAN, scale=RND_STD, size=[input_cnt, output_cnt])  # 가중치 초기화(정규분포)
    # Q: 편향 차원 왜 output_cnt 인가
    bias = np.zeros([output_cnt])  # output cnt 만큼 편향 초기화 
    
    return None

# 트레인
def train_and_test(epoch_cnt, mb_size, report):

    step_cnt = arange_data(mb_size)  # 데이터 섞기, train, test 분리, ?
    test_x, test_y = get_test_data()  # 테스트데이터 추출
    
    for epoch in range(epoch_cnt):
        # 에포크 마큼 반복
        losses, accs = [], []
        
        for n in range(step_cnt):
            train_x, train_y = get_train_data(mb_size, n)  # 미니배치 추출
            loss, acc = run_train(train_x, train_y)
            print(f'==={acc}===')
            losses.append(loss)
            accs.append(acc)
            
        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)

            # 에포크, 손실값평균, 정확도평균 출력
            print(f' Epoch {epoch+1}: loss={np.mean(losses)}, accuracy={np.mean(accs)}/{acc}')
            
    final_acc = run_test(test_x, test_y)
    print(f'Final test: final accuracy = {final_acc}')

    
# 데이터 섞기, 미니배치 생성, 테스트 시작 인덱스 지정#
def arange_data(mb_size):
    
    global data, shuffle_map, test_begin_idx
    
    shuffle_map = np.arange(data.shape[0])  # 미니배치 수만큼 인덱스 배열 생성
    np.random.shuffle(shuffle_map)  # 섞기
    step_cnt = int(data.shape[0] * 0.8) // mb_size  # 에포크당 배치 실행 수(20% 테스트셋)
    test_begin_idx = step_cnt * mb_size  # 테스트 시작 인덱스 지정
    
    return step_cnt


# 테스트데이터 추출
def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
   
    test_data = data[shuffle_map[test_begin_idx:]]
    test_x, test_y = test_data[:, :-output_cnt], test_data[:, -output_cnt:]
    return test_x, test_y
    

# 트레인데이터 추출
def get_train_data(mb_size, nth):
    
    global data, shuffle_map, test_begin_idx, output_cnt
    
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])  # 첫 수행일 때 섞기
        
    train_data = data[shuffle_map[mb_size*nth: mb_size*(nth+1)]]  # 미니배치 구간
    train_x, train_y = train_data[:, :-output_cnt], train_data[:, -output_cnt:]
    
    return train_x, train_y


# 학습 실행
def run_train(x, y):
    """미니배치 x,y 로 학습
    한 스텝만큼 학습
    전파 -> 결과 추출 -> 결과저장 순
    """
    output, aux_nn = forward_neuralnet(x)  # 순전파
    loss, aux_pp = forward_postproc(output, y)
    accuracy = forward_postproc(output, y)  # 결과 저장
    
    G_loss = 1  # 역전파  dL/dL  1
    G_output = backprop_postproc(G_loss, aux_pp)  # 역전파(역순)
    backprop_neuralnet(G_output, aux_nn)
    
    return loss, accuracy
                  
    
def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy

# 순전파, 역전파 함수 정의
def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x

def backprop_neuralnet(G_output, x):
    """역전파
    
    Parameters
    ----------
    G_ouput: float
        손실 기울기
    x: np.array
        
    """
    global weight, bias
    g_output_w = x.transpose()
    
    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)
    
    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b
    
# 루처리
def forward_postproc(output, y):
    """
    
    """
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def backprop_postproc(G_loss, diff):
    shape = diff.shape
    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1
    
    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff
    
    return G_output


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output-y) / y))
    return 1 - mdiff