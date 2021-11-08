# Brain-Tumor-Segmentation-Paddle


## 1.Introduction
This project is based on the paddlepaddle_V2.1 framework to reproduce Brain-Tumor-Segmentation.  

We put our project in paddle-bts/.
## 2.Result

The model is trained on the train set of BraTS2015, and we test it on the 0027 image(an image has 240 slices) as the author did.

Average result in all 240 slices:

 Version | Dice Complete | Dice Core | Dice Enhancing
 ---- | ----- | -----  | -----
 paddle version(ours) | 0.907|  0.961 | 1.0
 
 Result in slice 113:
 
  Version | Dice Complete | Dice Core | Dice Enhancing
 ---- | ----- | -----  | -----
 paddle version(ours) | 0.828|  0.935 | 1.0
 

The model file of the paddle version we trained：

Link: https://pan.baidu.com/s/1M5wGRSbIcmQLsvCoeS5Lsg password: pi6p


## 3.Requirements

 * Hardware：GPU（Tesla V100-32G is recommended）
 * Framework:  PaddlePaddle >= 2.1.2


## 4.Quick Start

### Step1: Clone

``` 
git clone https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation.git
cd paddle-bts
``` 

### Step2: Training

```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3: Evaluating

```  
CUDA_VISIBLE_DEVICES=0 python test.py # test in one slice
CUDA_VISIBLE_DEVICES=0 python test_all.py # test in all 240 slices
```  
