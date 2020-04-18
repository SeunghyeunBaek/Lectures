# Function details

## Neuralnet Architecture





## Flowchart

```flow
start=>start: 시작
load_data=>operation: 데이터불러오기
init_model=>operation: 가중치초기화
set_batch=>operation: 배치크기설정
shuffle=>operation: 데이터섞기
get_test_data=>operation: 테스트데이터분리

get_train_batch=>operation: 학습배치추출
foward_nn=>operation: 순전파(출력값)
foward_postproc=>operation: 순전파후처리(loss)
backprop_postproc=>operation: 역전파후처리(G_loss)
backprop_nn=>operation: 역전파(가중치업데이트)

is_max_epoch=>condition: 마지막에포크?
is_max_step=>condition:  마지막배치?

end=>end: 끝

start->load_data->init_model->set_batch->shuffle->get_test_data->get_train_batch->foward_nn->foward_postproc->backprop_postproc->backprop_nn->is_max_step

is_max_step(yes)->is_max_epoch
is_max_step(no)->get_train_batch

is_max_epoch(yes)->end
is_max_epoch(no)->shuffle
```





## Call graph

```mermaid
stateDiagram
[*] --> abalone_exec
abalone_exec --> load_abalone_dataset
abalone_exec --> init_model
abalone_exec --> train_and_test
train_and_test --> arrange_data
train_and_test --> get_train_data
train_and_test --> get_test_data
train_and_test --> run_train
train_and_test --> run_test

run_train --> forward_neuralnet
run_train --> forward_postproc
run_train --> backprop_neuralnet
run_train --> backprop_postproc
run_train --> eval_accuracy

run_test --> eval_accuracy
eval_accuracy --> [*]
```

## Variable List

* abalone_exec
* load_ablaone_dataset
* init_model
* train_and_test
* arrage_data
* get_train_data
* get_test_data
* run_train
* run_test
* forward_neuralnet
* forward_postproc
* eval_accuracy
* backprop_neuralnet
* backprop_postproc

