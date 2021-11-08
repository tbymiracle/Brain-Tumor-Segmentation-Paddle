# import keras
# from keras import layers
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add
# from keras.models import Model

import paddle
from paddle import nn
import numpy as np

class two_path(paddle.nn.Layer):
    def __init__(self,in_c):
        super(two_path, self).__init__()

        self.conv1_1 = nn.Conv2D(in_channels=in_c,out_channels=64, kernel_size=(7, 7), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn1_1 = nn.BatchNorm2D(64, data_format='NHWC')

        self.conv1_2 = nn.Conv2D(in_channels=in_c,out_channels=64, kernel_size=(7, 7), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn1_2 = nn.BatchNorm2D(64, data_format='NHWC')

        self.conv2 = nn.Conv2D(in_channels=64,out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.relu2 = nn.ReLU()

        self.conv3_1 = nn.Conv2D(in_channels=in_c,out_channels=160, kernel_size=(13, 13), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn3_1 = nn.BatchNorm2D(160, data_format='NHWC')
        self.conv3_2 = nn.Conv2D(in_channels=in_c,out_channels=160, kernel_size=(13, 13), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn3_2 = nn.BatchNorm2D(160, data_format='NHWC')

        self.conv4_1 = nn.Conv2D(in_channels=64,out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn4_1 = nn.BatchNorm2D(64, data_format='NHWC')
        self.conv4_2 = nn.Conv2D(in_channels=64,out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn4_2 = nn.BatchNorm2D(64, data_format='NHWC')

        self.conv5 = nn.Conv2D(in_channels=64,out_channels=64, kernel_size=(2, 2), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.relu5 = nn.ReLU()


    def forward(self, x_input):
        x = self.conv1_1(x_input)
        x = self.bn1_1(x)
        x1 = self.conv1_2(x_input)
        x1 = self.bn1_2(x1)
        x = paddle.maximum(x,x1)
        x = self.conv2(x)
        x = self.relu2(x)
        x2 = self.conv3_1(x_input)
        x2 = self.bn3_1(x2)
        x21 = self.conv3_2(x_input)
        x21 = self.bn3_2(x21)
        x2 = paddle.maximum(x2, x21)
        x3 = self.conv4_1(x)
        x3 = self.bn4_1(x3)
        x31 = self.conv4_2(x)
        x31 = self.bn4_2(x31)
        x = paddle.maximum(x3, x31)
        x = self.conv5(x)
        x = self.relu5(x)
        x = paddle.concat(x=[x2, x],axis=-1)

        return x

class input_cascade(paddle.nn.Layer):
    def __init__(self, input_shape1, input_shape2):
        super(input_cascade, self).__init__()
        self.Two_path1 = two_path(4)
        self.conv1 = nn.Conv2D(in_channels=224,out_channels=5, kernel_size=(21, 21), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2D(5, data_format='NHWC')
        self.Two_path2 = two_path(9)

        self.conv2 = nn.Conv2D(in_channels=224,out_channels=5, kernel_size=(21, 21), stride=(1, 1), padding='VALID', data_format='NHWC')
        self.bn2 = nn.BatchNorm2D(5, data_format='NHWC')
        self.softmax = nn.Softmax()



    def forward(self, x1_input,x2_input):

        # 1st two-path of cascade
        x1 = self.Two_path1(x1_input)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.bn1(x1)
        x2_input1 = paddle.concat([x1, x2_input],axis=-1)
        print(x2_input1)
        # X2_input1 = Input(tensor = X2_input1)
        x2 = self.Two_path2(x2_input1)
        print(x2)
        # Fully convolutional softmax classification
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.softmax(x2)
        return x2
