# CNN architectures

*AlexNet, VGG, GoogleLeNet, ResNet*

* 참고자료
  * [ ImageNet Classification with Deep Convolutional Neural Networks(2012)](https://github.com/sjchoi86/dl_tutorials_10weeks/blob/master/papers/ImageNet Classification with Deep Convolutional Neural Networks.pdf)
  * [Going Deeper with Convolutions(2015)](https://github.com/sjchoi86/dl_tutorials_10weeks/blob/master/papers/Going Deeper with Convolutions.pdf)
  * [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning(2016)](https://arxiv.org/pdf/1602.07261.pdf)

## 1. AlexNet

![](https://github.com/zzsza/Deep_Learning_starting_with_the_latest_papers/blob/master/Lecture_Note/images/cnn05.png?raw=true)

* 검토사항
  1. 각 layer를 정의하기 위한 parameter의 수
     * filter: 11 x 11 x 3 x 48
  2. 왜 layer가 2개로 나누어져있는가
     * GPU 메모리가 부족해서 두개로 나눔
* Relu
  * 왠만하면 분류에서는 Relu를 쓴다.
* LRN( Local Response Normalization )
  * 일정 부분만 activation 시킴
  * 활성화된 신경 주변의 신경은 비활성화
* Regularization
  * Data augmentation
    * 데이터를 늘리는 작업
    * 좌우반전(Flip), 잘라서사용(Crop + Flip)
      * 지금 찾고 싶은 물체의 종류에 따라 사용여부가 갈림
      * mnist에서는 사용할 수 없음: 숫자 6을 돌리면 6이 아님 ( Label preserve )
  * Color variation
    * 색상을 조정하는 작업
    * 학습을 통해 가장 많이 변화한 색상을 뽑아내고 그것에 비례해서 noise를 넣는다.
  * Dropout

## 2. VGG

## 3. GoogLeNet

* Google + Lenet ( 22 layers )
* **1 x 1 convolution for dimension reduction** + **갈림길**
* VGG 보다 깊지만 파라메터수가 거의 절반임

### 3-1. Inception Module

[그림 - Naive inception Module]

* 1 x 1 convolution layer
  * Inception module = Naive inception module + 1x1 convolution( dimension reduction)

![dimension reduction](https://github.com/zzsza/Deep_Learning_starting_with_the_latest_papers/blob/master/Lecture_Note/images/cnn07.png?raw=true)

* 1 x 1 convolution 으로 파라메터수를 줄였음 

* **input layer 의 channel 수를 줄여서 parameter 수를 줄일 수 있었다.**

* **하나의 데이터를 쪼개서 각각 filter를 씌우고 나중에 합친다( 갈림길 )**

  ![](https://www.researchgate.net/publication/318853350/figure/fig2/AS:631664187285522@1527611886306/GoogleNet-Inception-V1.png)

  * receptive field : 출력단의 픽셀이 입력단의 얼마나 많은 픽셀을 담고있는가
    * 항상 동일한 크기의 영역만 정보를 압축함
  * receptive field를 다양하게 학습시켜 정보를 취합한다. 

### 3-2. Inception V4

![](https://miro.medium.com/max/3702/1*LjImFrHzbu2mG8RezRO_JA.png)

* 동일한 receptive field를 얻으면서 parameter를 줄이는 것이 목적
  * ex) 3x3 filter 2번 했을 때 parameter 는 18개, 5x5 filter 는 25개. 둘다 receptive field는 동일하다
* 1x7 - 7 x 1 convolution 

## 4. ResNet

* 152 layers, 많은 대회에서 우승 => 범용적으로 사용할 수 있음

* 문제
  * is deeper network always better?
  * what about vanishing / exploding gradients?
    * => better initialization method + batch normalization + active function 으로 해결
  * no overfitting, but degradation 
    * overfitting: train 데이터만 잘 맞춤
    * **degradation problem**: train, test 둘다 성능이 좋은데 깊어질수록 성능이 잘 안나옴

### 4-1. Residual learning building block

![](https://www.researchgate.net/profile/Syeda_Atik2/publication/330880103/figure/fig5/AS:723733501194240@1549562921364/A-building-block-of-Residual-learning.png)

* 입력을 출력에 더함( 입력과 출력의 dimension이 같아야함 )
* 머신은 residual만 학습함
  * 입력과 출력의 차이만 학습( 코드)

### 4-2. 파라메터 줄이기

![](https://upload-images.jianshu.io/upload_images/2228224-f9b16ae3483b14da.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

* deeper bottle-neck

* 1x1 -> convolution -> 1x1(원래 input size로 복원)

### 4-3. 결론 및 한계

* 이전까지 40단에서 발생하던 degradation을 100단 이후로 넘김. 여전히 1000단 이상으로 넘어가면 degradation 발생.

