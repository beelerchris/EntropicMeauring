import matplotlib.pyplot as plt
import numpy as np

data = np.load('pixel_dist.npz')
dist0 = data['dist0']
dist1 = data['dist1']

for i in range(10):
    _plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(10, 8))
    _plot_lines = []
    mappable = _plot_axs.pcolormesh(dist0[i, ::-1], vmin=0, vmax=1, cmap='summer')
    _plot_fig.colorbar(mappable)
    _plot_axs.set_xticks([])
    _plot_axs.set_yticks([])
    _plot_axs.set_xlim([0, dist0.shape[1]])
    _plot_axs.set_ylim([0, dist0.shape[2]])
    plt.savefig('./dist_im/dist0_%d.png' % (i))
    plt.close()

for i in range(10):
    _plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(10, 8))
    _plot_lines = []
    mappable = _plot_axs.pcolormesh(dist1[i, ::-1], vmin=0, vmax=1, cmap='summer')
    _plot_fig.colorbar(mappable)
    _plot_axs.set_xticks([])
    _plot_axs.set_yticks([])
    _plot_axs.set_xlim([0, dist1.shape[1]])
    _plot_axs.set_ylim([0, dist1.shape[2]])
    plt.savefig('./dist_im/dist1_%d.png' % (i))
    plt.close()
