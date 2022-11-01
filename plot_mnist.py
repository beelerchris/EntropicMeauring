from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_labels) = mnist.load_data()
sample = train_x[np.random.randint(0, train_x.shape[0])]

_plot_fig, _plot_axs = plt.subplots(1, 2, figsize=(16, 8))
_plot_lines = []
mappable = _plot_axs[0].pcolormesh(sample[::-1] / 255, vmin=0, vmax=1, cmap='gray')
#_plot_fig.colorbar(mappable)
_plot_axs[0].set_xticks([])
_plot_axs[0].set_yticks([])
_plot_axs[0].set_xlim([0, sample.shape[0]])
_plot_axs[0].set_ylim([0, sample.shape[1]])

mappable = _plot_axs[1].pcolormesh(sample[::-1] // 128, vmin=0, vmax=1, cmap='gray')
#_plot_fig.colorbar(mappable)
_plot_axs[1].set_xticks([])
_plot_axs[1].set_yticks([])
_plot_axs[1].set_xlim([0, sample.shape[0]])
_plot_axs[1].set_ylim([0, sample.shape[1]])
plt.savefig('./dist_im/mnist_sample.png', bbox_inches='tight')
plt.close()

