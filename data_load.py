import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize


def model_gen(input_dim, x, y, slice_no):
    X1 = []
    X2 = []
    Y = []
    u = (y[:, slice_no, :])
    print(y[1, 113, 1])
    for i in range(int((input_dim) / 2), y.shape[0] - int((input_dim) / 2)):
        for j in range(int((input_dim) / 2), y.shape[2] - int((input_dim) / 2)):
            # Filtering all 0 patches
            if (x[i - 16:i + 17, j - 16:j + 17, :].any != 0):
                X2.append(x[i - 16:i + 17, j - 16:j + 17, :])
                X1.append(x[i - int((input_dim) / 2):i + int((input_dim) / 2) + 1,
                          j - int((input_dim) / 2):j + int((input_dim) / 2) + 1, :])
                Y.append(y[i, slice_no, j])

    X1 = np.asarray(X1, 'f')
    X2 = np.asarray(X2, 'f')
    Y = np.asarray(Y)
    d = [X1, X2, Y]
    return d


def data_gen(data, y, slice_no, model_no):
    d = []
    x = data[slice_no]
    # filtering all 0 slices and non-tumor slices
    if (x.any() != 0 and y.any() != 0):
        if (model_no == 0):
            X1 = []
            for i in range(16, 138):
                for j in range(16, 223):
                    if (x[i - 16:i + 17, j - 16:j + 17, :].all != 0):
                        X1.append(x[i - 16:i + 17, j - 16:j + 17, :])
            Y1 = []
            for i in range(16, 138):
                for j in range(16, 223):
                    if (x[i - 16:i + 17, j - 16:j + 17, :].all != 0):
                        Y1.append(y[i, slice_no, j])
            X1 = np.asarray(X1)
            Y1 = np.asarray(Y1)
            d = [X1, Y1]
        elif (model_no == 1):
            d = model_gen(65, x, y, slice_no)
        elif (model_no == 2):
            d = model_gen(56, x, y, slice_no)
        elif (model_no == 3):
            d = model_gen(53, x, y, slice_no)

    return d


class BraTsDataset(Dataset):
    """
    数据集定义
    """

    def __init__(self, x1, x2, y):
        print(x1.shape, x2.shape, y.shape)
        self.train_images1 = x1
        self.train_images2 = x2
        self.label_images = y

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        train_image1 = self.train_images1[idx]
        train_image2 = self.train_images2[idx]
        train_image1 = np.array(train_image1, dtype='float32')
        train_image2 = np.array(train_image2, dtype='float32')
        label_image = self.label_images[idx]
        label_image = np.array(label_image)
        return train_image1, train_image2, label_image

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.train_images1)