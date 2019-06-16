"""
This file is copied from yscbm's Github: yscbm8@gmail.com

https://github.com/yscbm/tensorflow/blob/master/common/extract_cifar10.py
"""
import gzip
import numpy as np
import os

import tensorflow as tf

LABEL_SIZE = 1
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_CLASSES = 10

TRAIN_NUM = 10000
TRAIN_NUMS = 50000
TEST_NUM = 10000


def extract_data(filenames):
    # 验证文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 读取数据
    labels = None
    images = None
    # 是否正在读取第一个文件
    flag = True

    for f in filenames:
        bytestream = open(f, 'rb')
        # 读取数据
        buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_SIZE))
        # 把数据流转化为np的数组
        data = np.frombuffer(buf, dtype=np.uint8)
        # 改变数据格式
        data = data.reshape(TRAIN_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)  # data的每个图像，label在前，image在后
        # 分割数组
        labels_images = np.hsplit(data, [LABEL_SIZE])

        label = labels_images[0].reshape(TRAIN_NUM)
        image = labels_images[1].reshape(TRAIN_NUM, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

        if flag:
            labels = label
            images = image
            flag = False
        else:
            # 合并数组，不能用加法
            labels = np.concatenate((labels, label))
            images = np.concatenate((images, image))
        pass

    # 数据处理，便于计算
    images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    return labels, images


def extract_train_data(files_dir):
    # 获得训练数据
    filenames = [os.path.join(files_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    return extract_data(filenames)


def extract_test_data(files_dir):
    # 获得测试数据
    filenames = [os.path.join(files_dir, 'test_batch.bin')]
    return extract_data(filenames)


# 把稠密数据label[1,5...]变为[[0,1,0,0...],[...]...]
def dense_to_one_hot(labels_dense, num_classes):
    # 数据数量
    num_labels = labels_dense.shape[0]
    # 生成[0,1,2...]*10,[0,10,20...]
    index_offset = np.arange(num_labels) * num_classes
    # 初始化np的二维数组
    labels_one_hot = np.zeros((num_labels, num_classes))
    # 相对应位置赋值变为[[0,1,0,0...],[...]...]
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class Cifar10DataSet(object):
    """docstring for Cifar10DataSet"""

    def __init__(self, data_dir, one_hot=True):
        super(Cifar10DataSet, self).__init__()
        self.train_labels, self.train_images = extract_train_data(
            os.path.join(data_dir, 'cifar-10-batches-bin'))
        self.test_labels, self.test_images = extract_test_data(os.path.join(data_dir, 'cifar-10-batches-bin'))

        # print(self.train_labels.size)

        if one_hot:
            self.train_labels = dense_to_one_hot(self.train_labels, NUM_CLASSES)
            self.test_labels = dense_to_one_hot(self.test_labels, NUM_CLASSES)

        # epoch完成次数
        self.epochs_completed = 0
        # 当前批次在epoch中进行的进度
        self.index_in_epoch = 0

    def next_train_batch(self, batch_size):
        # 起始位置
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        # print "self.index_in_epoch: ",self.index_in_epoch
        # 完成了一次epoch
        if self.index_in_epoch > TRAIN_NUMS:
            # epoch完成次数加1
            self.epochs_completed += 1
            # print "self.epochs_completed: ",self.epochs_completed
            # 打乱数据顺序，随机性
            perm = np.arange(TRAIN_NUMS)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            start = 0
            self.index_in_epoch = batch_size
            # 断言：条件不成立会报错
            assert batch_size <= TRAIN_NUMS

        end = self.index_in_epoch
        # print "start,end: ",start,end

        return self.train_images[start:end], self.train_labels[start:end]

    def test_data(self):
        return self.test_images, self.test_labels

    def train_data(self):
        return self.train_images, self.train_labels


def main():
    # train_labels, train_images = extract_train_data('./data/cifar-10-batches-bin')
    # print(train_images.shape)
    cc = Cifar10DataSet('./data/')
    cc.next_train_batch(100)


if __name__ == '__main__':
    main()
