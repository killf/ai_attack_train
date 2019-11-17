import numpy as np
from scipy.io import loadmat
import os, cv2


class DataSet:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.train_list = loadmat(os.path.join(data_dir, "train_list.mat"))
        self.test_list = loadmat(os.path.join(data_dir, "test_list.mat"))

        self.train_length = self.train_list['labels'].size
        self.test_length = self.test_list['labels'].size

    def load_image(self, file):
        img = cv2.imread(file)
        img = cv2.resize(img, (244, 224))
        img = np.transpose(img, [2, 0, 1]).astype('float32')
        return img / 127.5 - 1.0

    def train_data(self):
        for i in range(self.train_length):
            file = self.train_list['file_list'][i][0][0]
            label = self.train_list['labels'][i][0]

            file = os.path.join(self.data_dir, "Images", file)

            yield self.load_image(file), label

    def test_data(self):
        for i in range(self.test_length):
            file = self.test_list['file_list'][i][0][0]
            label = self.test_list['labels'][i][0]

            file = os.path.join(self.data_dir, "Images", file)

            yield self.load_image(file), label


# HWC
# NCHW
if __name__ == '__main__':
    dataset = DataSet("/home/killf/dataset/ai_attack")
    for img, label in dataset.train_data():
        cv2.imshow("", img)
        cv2.waitKey()
        print(img, label)
        exit()
