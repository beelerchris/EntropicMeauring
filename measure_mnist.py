from copy import deepcopy
import numpy as np
from keras.datasets import mnist
import warnings

warnings.filterwarnings(action='ignore', category=RuntimeWarning)

def entropy_calc(samples, s_labels):
    dist0 = np.zeros((s_labels.max()+1, samples.shape[1], samples.shape[2]), dtype=np.float64)
    dist1 = np.zeros((s_labels.max()+1, samples.shape[1], samples.shape[2]), dtype=np.float64)

    count = np.zeros(s_labels.max()+1, dtype=np.int32)
    for i in range(samples.shape[0]):
        dist0[s_labels[i]] += (1 - samples[i])
        dist1[s_labels[i]] += samples[i]
        count[s_labels[i]] += 1

    dist = dist1.sum(axis=0) / samples.shape[0]
    dist0 /= dist0.sum(axis=0)
    dist1 /= dist1.sum(axis=0)

    H0 = -1.0 * np.nan_to_num(dist0 * np.log(dist0)).sum(axis=0)
    H1 = -1.0 * np.nan_to_num(dist1 * np.log(dist1)).sum(axis=0)

    H = (1 - dist) * H0 + dist * H1

    return H, count / count.sum()

(train_x, train_y), (test_x, test_labels) = mnist.load_data()

all_data = np.zeros((train_x.shape[0] + test_x.shape[0], train_x.shape[1], train_x.shape[2]), dtype=np.int32)

all_data[:train_x.shape[0]] = np.array(train_x) // 128
all_data[train_x.shape[0]:] = np.array(test_x) // 128
all_labels = np.array(list(train_y) + list(test_labels))

state = np.zeros((2, all_data.shape[1], all_data.shape[2]), dtype=np.int32)
sample = all_data[np.random.randint(0, all_data.shape[0])]
count = np.zeros(1)

i = -1
while count.max() < 0.95:
    i += 1
    masked_inds = np.all(((all_data * state[0]) == state[1]) == True, axis=(1, 2))
    data = all_data[masked_inds]
    labels = all_labels[masked_inds]

    H, count = entropy_calc(data, labels)
    ind = np.unravel_index(H.argmin(), H.shape)

    state[0][ind] = 1
    state[1][ind] = deepcopy(sample[ind])

    print("Measurement: %d\nClassification: %d\nConfidence: %.4f\n" % (i, count.argmax(), count.max()))

masked_inds = np.all(((all_data * state[0]) == state[1]) == True, axis=(1, 2))
data = all_data[masked_inds]
labels = all_labels[masked_inds]

H, count = entropy_calc(data, labels)
print("Measurement: %d\nClassification: %d\nConfidence: %.4f" % (i+1, count.argmax(), count.max()))
