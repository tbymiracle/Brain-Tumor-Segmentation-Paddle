import os
from sklearn.utils import class_weight
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from model import two_path, input_cascade
from data_load import data_gen,BraTsDataset
import paddle
from paddle import nn
from paddle.static import InputSpec
import paddle.nn.functional as F
from sklearn import metrics


m = input_cascade((65,65,4),(33,33,4))

input = [InputSpec([None, 65, 65, 4], 'float32', 'image'),InputSpec([None, 33, 33, 4], 'float32', 'image1')]
label = InputSpec([None, 1,1,5], 'float32', 'label')

model = paddle.Model(m,input,label)
model.prepare()
model.load('bts_paddle1.pdparams')
print("hahasa")


path = '../HGG'
p = os.listdir(path)
p.sort(key=str.lower)
arr = []
qq = 'brats_2013_pat0027_1'
q = os.listdir(path + '/' + qq)
q.sort(key=str.lower)
print(q)
for j in range(len(q)):
    if (j != 4):
        img = sitk.ReadImage(path + '/' + qq + '/' + q[j])
        arr.append(sitk.GetArrayFromImage(img))
    else:
        # print(q[j])
        img = sitk.ReadImage(path + '/' + qq + '/' + q[j])
        Y_labels = sitk.GetArrayFromImage(img)
        print(Y_labels.shape)
data = np.zeros((Y_labels.shape[1], Y_labels.shape[0], Y_labels.shape[2], 4))
for i in range(Y_labels.shape[1]):
    data[i, :, :, 0] = arr[0][:, i, :]
    data[i, :, :, 1] = arr[1][:, i, :]
    data[i, :, :, 2] = arr[2][:, i, :]
    data[i, :, :, 3] = arr[3][:, i, :]
print(data.shape)
info = []
# Creating patches for each slice and training(slice-wise)
print(data.shape[0])
#test
d_test = data_gen(data,Y_labels,113,1)
print(Y_labels[0])
if(len(d_test) != 0):
    y_test = np.zeros((d_test[2].shape[0],1,1,5))
    for j in range(y_test.shape[0]):
        y_test[j,:,:,d_test[2][j]] = 1
    X1_test = d_test[0]
    X2_test = d_test[1]
    y_test1 = np.argmax(y_test,axis = -1)
    testset = BraTsDataset(X1_test,X2_test,y_test1)
    print(X1_test.shape,X2_test.shape,y_test1.shape)
    pred = model.predict(testset,batch_size = 64)
    print(pred)
    print(len(pred[0]),pred[0][0].shape)
    pred_fi = np.concatenate(pred[0],axis=0)
    pred_fi = np.around(pred_fi)
    pred1 = np.argmax(pred_fi.reshape(pred_fi.shape[0],5)[:,1:4],axis = 1)
    y2 = np.argmax(y_test.reshape(y_test.shape[0],5)[:,1:4],axis = 1)
    print(pred1[0])
    print(y2[0])
    f1 = metrics.f1_score(y2,pred1,average='micro')
    print(f1)

