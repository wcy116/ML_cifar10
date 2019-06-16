import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import tensorflow.examples.tutorials.mnist as mn

import extract_cifar10
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.utils import shuffle


def main():
    # # # ------------ PCA on Cifar-10 dataset ------------ # # #
    # # Loading the dataset.
    cifar10_data_set = extract_cifar10.Cifar10DataSet('./data/', one_hot=False)
    train_images, train_labels = cifar10_data_set.train_data()
    test_images, test_labels = cifar10_data_set.test_data()

    train_images, train_labels = shuffle(train_images, train_labels)

    X = train_images.reshape((50000, 32*32*3))[:15000, :]
    y = train_labels[:15000]

    # # Colors and Markers
    colors = {0: '#FF5733',
              1: '#FCFF33',
              2: '#4CFC1C',
              3: '#09FEE2',
              4: '#D912D9',
              5: '#000000',
              6: '#125FD9',
              7: '#2F9A5F',
              8: '#CBFF33',
              9: '#FFA833'}

    markers = {0: '.',
               1: 'v',
               2: '^',
               3: '<',
               4: '>',
               5: 's',
               6: 'p',
               7: 'P',
               8: '+',
               9: 'x'}

    # # Standardizing
    X_std = StandardScaler().fit_transform(X)

    plot_in_2D = False

    if plot_in_2D:
        # # PCA using sklearn
        sklearn_pca = sklearnPCA(n_components=2)
        Y_sklearn = sklearn_pca.fit_transform(X_std)

        # # Plot the result in 2D.
        data = []
        for lab in range(10):
            trace = dict(
                label=lab,
                data=Y_sklearn[y == lab, :]
            )
            data.append(trace)

        for i in range(len(data)):
            data_x = data[i]['data'][:, 0].ravel()
            data_y = data[i]['data'][:, 1].ravel()
            plt.scatter(data_x, data_y, c=colors[data[i]['label']], label=data[i]['label'],
                        marker=markers[data[i]['label']])

        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend(loc='best')
        plt.show()
    else:
        # # PCA using sklearn
        sklearn_pca = sklearnPCA(n_components=3)
        Y_sklearn = sklearn_pca.fit_transform(X_std)

        # Plot the result in 3D.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        data = []
        for lab in range(10):
            a = Y_sklearn[y == lab, :]
            trace = dict(
                label=lab,
                data=Y_sklearn[y == lab, :]
            )
            data.append(trace)

        for i in range(len(data)):
            label = data[i]['label']
            data_x = data[i]['data'][:, 0].ravel()
            data_y = data[i]['data'][:, 1].ravel()
            data_z = data[i]['data'][:, 2].ravel()
            ax.scatter(data_x, data_y, data_z, c=colors[label], label=label, marker=markers[label])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    main()
