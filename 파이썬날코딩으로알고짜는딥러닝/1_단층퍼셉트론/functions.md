# Function details

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

## Function list

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

