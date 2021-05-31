# Superresolution using an efficient sub-pixel convolutional neural network

EPSCN_tensorflow 레파지토리는 colab 공식 espcn 코드를 활용하여 작성하였습니다.   
["Colab code"]("https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/super_resolution_sub_pixel.ipynb#scrollTo=il1NiNVcOAuA") <- 클릭 시 코랩 코드로 이동
   
해당 코드는 tensorflow를 활용하여 작성되었습니다.
["super_resolution_sub_pixel"]("https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/super_resolution_sub_pixel.ipynb")

## EPSCN을 활용한 Super-Resolution

논문:
["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158)  
해당 논문은 네트워크 안에서 초해상화와 같은 역할을 수행하기 위해 공간 해상도를 높이는 방법으로 ESPCN이라는 새로운 방법을 제안하였습니다.

```
#train
python train.py

#test
python test.py
```
