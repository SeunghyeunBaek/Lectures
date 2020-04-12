# CNN 기초

![cnn_concept](https://github.com/zzsza/Deep_Learning_starting_with_the_latest_papers/blob/master/Lecture_Note/images/cnn01.png?raw=true)

* CNN
  * CNN = feature extractor + classifier
  * feature extractor = Convolution + subsampling
  * classifier = fully connected layers

* CNN이 잘 동작하는 이유
  1. Local Invariance
     * convolution filter 가 이미지 내 모든 픽셀을 훑는다.
     * 이미지 내 물체가 어디에 있던간에 찾을 수 있다.
  2. Compositionality
     * 계층구조

## 1. Convolution

* Convolution Filter의 값과 Convolution Filter가 위치한 부분의 픽셀값을 곱한 후 모두 더하는 과정
* 결과값은 해당 부분과 Convolution Filter의 유사도를 의미한다.
* Convolution Filter 모양은 학습을 통해 결정된다.

### 1-1. Convolution - 용어

* Zero Padding
  * 이미지의 가장자리에서 convolution할 수 있도록 0을 채워넣음
  * Zero - padding 식
* Stride
  * convolution filter가 움직이는 단위

![cnn_convolution_filter_concept](https://github.com/zzsza/Deep_Learning_starting_with_the_latest_papers/blob/master/Lecture_Note/images/cnn03.png?raw=true)

* tensorflow parameters
  * ```tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)```
    * `input`
      - `batch`: 학습에 필요한 이미지 수
      - `in_height`, `out_height`: 이미지 크기
      - `in_chnnel`: rgb일 때 3
    * `filter`
      * `filter_height`, `filter_width`: 필터 크기
      * `in_channels`: rgb일 때 3, `input`의 `in_chnnel`과 같아야함
      * `out_channels`: convolution filter의 수
* number of parameters
  * **parameter가 적을 수록 좋은 성능을 낼 수 있음**
  * parameter 수를 줄이면서 많은 수의 레이어를 쌓아야함
  * filter 가 [3, 3, 3, 7] 일 때 parameters의 수는 189개
  * convolution layer에 비해 fully connect layer의 parameter 수가 급격하게 많아지기 때문에 이를 줄이는 방안을 생각해야함

## 2. CNN process

![cnn_process](https://github.com/zzsza/Deep_Learning_starting_with_the_latest_papers/blob/master/Lecture_Note/images/cnn04.png?raw=true)

1. convolution layer
   * 이미지: 28 x 28
   * filter: 3 x 3(filter size) x 64(n_filters)
2. bias add layer
   * 1 x 64
3. active function  layer
   * relu
4. pooling(max, average) layer 
   * max pooling(4 x 4)
5. fully connected layer
   * reshape input: 14 x 14 x 64 데이터를 1 x (14 x 14 x 64) vector로 변환
   * fully connected layer : input dimension => 1 x 10
   * output: 1x10

* 파라메터의 수
  * convolution layer: 3 x 3(filter size)x 64(n_filters) + 64(n_bias)  = 600
  * fully connected layer: 14 x 14(layer size after max pulling - 4 x 4) x 64(n_filters) x 10(n_bias = n_classes) = 120k
  * fully connected layer에서 parameter의 수가 급증한다.