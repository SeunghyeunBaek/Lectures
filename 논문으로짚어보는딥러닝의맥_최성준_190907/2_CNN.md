# CNN 기초

[그림 CNN 전체 구조]

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

[그림 Convolution 그림]

## Convolution

* Convolution Filter의 값과 Convolution Filter가 위치한 부분의 픽셀값을 곱한 후 모두 더하는 과정
* 결과값은 해당 부분과 Convolution Filter의 유사도를 의미한다.
* Convolution Filter 모양은 학습을 통해 결정된다.
* Zero Padding
  * 이미지의 가장자리에서 convolution할 수 있도록 가장자리에 0을 채워넣음
  * Zero - padding 식
* Stride
  * convolution filter가 움직이는 단위

[그림  - filter 그림]

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
* number of parameters?
  * parameter 수를 줄이면서 많은 수의 레이어를 쌓아야함
  * filter 가 [3, 3, 3, 7] 일 때 parameters의 수는 189개

[그림 CNN 과정]

* 각 단계별 parameter 수 계산해보기
* Fully connected 의 parameter를 작게 만들어야함 