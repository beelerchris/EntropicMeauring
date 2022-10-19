from copy import deepcopy
import numpy as np
from keras.datasets import mnist
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=RuntimeWarning)

def entropy_calc(samples, s_labels, n_labels):
    dist0 = np.zeros((n_labels, samples.shape[1], samples.shape[2]), dtype=np.float64)
    dist1 = np.zeros((n_labels, samples.shape[1], samples.shape[2]), dtype=np.float64)

    count = np.zeros(n_labels, dtype=np.int32)
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

    return H, dist1, count / count.sum()

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

    H, dist1, count = entropy_calc(data, labels, all_labels.max()+1)
    ind = np.unravel_index(H.argmin(), H.shape)

    print("Measurement: %d\nClassification: %d\nConfidence: %.4f\n" % (i, count.argmax(), count.max()))

    _plot_fig, _plot_axs = plt.subplots(2, 7, figsize=(36, 8))
    _plot_lines = []
    plot_state = ((state[1] + 0.5) - state[0] * 0.5)
    mappable = _plot_axs[0, 0].pcolormesh(plot_state[::-1], vmin=0, vmax=1, cmap='gray')
    _plot_axs[0, 0].set_xticks([])
    _plot_axs[0, 0].set_yticks([])
    _plot_axs[0, 0].set_xlim([0, state.shape[1]])
    _plot_axs[0, 0].set_ylim([0, state.shape[2]])

    mappable = _plot_axs[0, 1].pcolormesh(sample[::-1], vmin=0, vmax=1, cmap='gray')
    _plot_axs[0, 1].set_xticks([])
    _plot_axs[0, 1].set_yticks([])
    _plot_axs[0, 1].set_xlim([0, sample.shape[0]])
    _plot_axs[0, 1].set_ylim([0, sample.shape[1]])

    mappable = _plot_axs[1, 0].pcolormesh(H[::-1], vmin=0, vmax=H.max(), cmap='summer')
    _plot_axs[1, 0].set_xticks([])
    _plot_axs[1, 0].set_yticks([])
    _plot_axs[1, 0].set_xlim([0, H.shape[0]])
    _plot_axs[1, 0].set_ylim([0, H.shape[1]])

    _plot_axs[1, 1].bar(np.arange(0, all_labels.max()+1, 1, dtype=np.int32), count)
    _plot_axs[1, 1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    _plot_axs[1, 1].set_ylim([0, 1])

    for j in range(10):
        mappable = _plot_axs[j//5, 2+(j%5)].pcolormesh(dist1[j, ::-1], vmin=0, vmax=1, cmap='gray')
        _plot_axs[0, 0].set_xticks([])
        _plot_axs[0, 0].set_yticks([])
        _plot_axs[0, 0].set_xlim([0, dist1.shape[1]])
        _plot_axs[0, 0].set_ylim([0, dist1.shape[2]])

    plt.savefig('./dist_im/EntMeas_ex_%.2d.png' % (i))
    plt.close()

    state[0][ind] = 1
    state[1][ind] = deepcopy(sample[ind])

masked_inds = np.all(((all_data * state[0]) == state[1]) == True, axis=(1, 2))
data = all_data[masked_inds]
labels = all_labels[masked_inds]

H, count = entropy_calc(data, labels, all_labels.max()+1)
print("Measurement: %d\nClassification: %d\nConfidence: %.4f" % (i+1, count.argmax(), count.max()))

_plot_fig, _plot_axs = plt.subplots(2, 7, figsize=(36, 8))
_plot_lines = []
plot_state = ((state[1] + 0.5) - state[0] * 0.5)
mappable = _plot_axs[0, 0].pcolormesh(plot_state[::-1], vmin=0, vmax=1, cmap='gray')
_plot_axs[0, 0].set_xticks([])
_plot_axs[0, 0].set_yticks([])
_plot_axs[0, 0].set_xlim([0, state.shape[1]])
_plot_axs[0, 0].set_ylim([0, state.shape[2]])

mappable = _plot_axs[0, 1].pcolormesh(sample[::-1], vmin=0, vmax=1, cmap='gray')
_plot_axs[0, 1].set_xticks([])
_plot_axs[0, 1].set_yticks([])
_plot_axs[0, 1].set_xlim([0, sample.shape[0]])
_plot_axs[0, 1].set_ylim([0, sample.shape[1]])

mappable = _plot_axs[1, 0].pcolormesh(H[::-1], vmin=0, vmax=H.max(), cmap='summer')
_plot_axs[1, 0].set_xticks([])
_plot_axs[1, 0].set_yticks([])
_plot_axs[1, 0].set_xlim([0, H.shape[0]])
_plot_axs[1, 0].set_ylim([0, H.shape[1]])

_plot_axs[1, 1].bar(np.arange(0, all_labels.max()+1, 1, dtype=np.int32), count)
_plot_axs[1, 1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
_plot_axs[1, 1].set_ylim([0, 1])

for j in range(10):
    mappable = _plot_axs[j//5, 2+(j%5)].pcolormesh(dist1[j, ::-1], vmin=0, vmax=1, cmap='gray')
    _plot_axs[0, 0].set_xticks([])
    _plot_axs[0, 0].set_yticks([])
    _plot_axs[0, 0].set_xlim([0, dist1.shape[1]])
    _plot_axs[0, 0].set_ylim([0, dist1.shape[2]])

plt.savefig('./dist_im/EntMeas_ex_%.2d.png' % (i))
plt.close()
