from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_labels) = mnist.load_data()
sample = train_x[np.random.randint(0, train_x.shape[0])]

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(10, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(sample[::-1], vmin=0, vmax=255, cmap='summer')
_plot_fig.colorbar(mappable)
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, sample.shape[0]])
_plot_axs.set_ylim([0, sample.shape[1]])
plt.savefig('./dist_im/mnist_sample.png')
plt.close()

plt.figure()
plt.imshow(sample, cmap='gray')
plt.show()
