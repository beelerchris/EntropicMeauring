from keras.datasets import mnist
import numpy as np

(train_x, train_y), (test_x, test_labels) = mnist.load_data()

data = np.zeros((train_x.shape[0] + test_x.shape[0], train_x.shape[1], train_x.shape[2]), dtype=np.int32)

data[:train_x.shape[0]] = np.array(train_x) // 128
data[train_x.shape[0]:] = np.array(test_x) // 128
labels = list(train_y) + list(test_labels)

dist0 = np.zeros((max(labels)+1, data.shape[1], data.shape[2]), dtype=np.float64)
dist1 = np.zeros((max(labels)+1, data.shape[1], data.shape[2]), dtype=np.float64)

for i in range(data.shape[0]):
    dist0[labels[i]] += (1 - data[i])
    dist1[labels[i]] += data[i]

dist = dist1.sum(axis=0) / data.shape[0]
dist0 /= dist0.sum(axis=0)
dist1 /= dist1.sum(axis=0)

np.savez('init_pixel_dist.npz', dist=dist, dist0=dist0, dist1=dist1)
