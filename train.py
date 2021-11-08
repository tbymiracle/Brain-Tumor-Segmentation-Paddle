import os
from sklearn.utils import class_weight
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from model import two_path, input_cascade
from data_load import data_gen, BraTsDataset
import paddle
from paddle import nn
from paddle.static import InputSpec
import paddle.nn.functional as F

# m0 = two_pathcnn((33,33,4))
# m0.summary()
#
# m1 = MFCcascade((53,53,4),(33,33,4))
# m1.summary()

m = input_cascade((65, 65, 4), (33, 33, 4))

input = [InputSpec([None, 65, 65, 4], 'float32', 'image'), InputSpec([None, 33, 33, 4], 'float32', 'image1')]
label = InputSpec([None, 1, 1, 5], 'float32', 'label')

model = paddle.Model(m, input, label)
#model.load('./bc/brats_2013_pat0001_1trial_0001_input_cascasde_acc')
path = '../HGG'
p = os.listdir(path)
p.sort(key=str.lower)
arr = []
print(p)
for k in range(len(p)):
    if k < 1:
        continue
    print(p[k])
    q = os.listdir(path + '/' + p[k])
    q.sort(key=str.lower)
    for j in range(len(q)):
        if (j != 4):
            img = sitk.ReadImage(path + '/' + p[k] + '/' + q[j])
            arr.append(sitk.GetArrayFromImage(img))
        else:
            # print(q[j])
            img = sitk.ReadImage(path + '/' + p[k] + '/' + q[j])
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
    for i in range(data.shape[0]):
        if (k == 1) and (i < 74):
            continue
        if i == 120:
            model.save('bc/' + p[k] + 'half_trial_0001_input_cascasde_acc')
        elif i == 239:
            model.save('bc/' + p[k] + 'trial_0001_input_cascasde_acc')

        d = data_gen(data, Y_labels, i, 1)
        if (len(d) != 0):
            y = np.zeros((d[2].shape[0], 1, 1, 5))
            for j in range(y.shape[0]):
                y[j, :, :, d[2][j]] = 1
            X1 = d[0]
            X2 = d[1]
            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(d[2]),
                                                              y=d[2])
            class_weights_final = np.zeros([5])
            for i_num, i_class in enumerate(np.unique(d[2])):
                class_weights_final[i_class] = class_weights[i_num]
            print(class_weights_final)
            class_weights_final = paddle.to_tensor(class_weights_final)
            #
            # class_weight_dict = dict(zip([x for x in np.unique(d[2])], class_weights))
            #
            # print(class_weight_dict)
            optim = paddle.optimizer.Adam(parameters=model.parameters())
            model.prepare(
                optim,
                nn.CrossEntropyLoss(weight=class_weights_final),
                paddle.metric.Accuracy())
            print('slice no:' + str(i))

            y = np.argmax(y, axis=-1)
            trainset = BraTsDataset(X1, X2, y)
            info.append(model.fit(trainset, epochs=1, batch_size=128))
            # info.append(m1.fit([X1, X2], y, epochs=1, batch_size=128))
            model.save('trial_0001_input_cascasde_acc')
            print("hello")

