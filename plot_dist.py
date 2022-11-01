import matplotlib.pyplot as plt
import numpy as np

data = np.load('init_pixel_dist.npz')
dist = data['dist']
dist0 = data['dist0']
dist1 = data['dist1']

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(10, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(dist[::-1], vmin=0, vmax=1, cmap='summer')
_plot_fig.colorbar(mappable)
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dist.shape[0]])
_plot_axs.set_ylim([0, dist.shape[1]])
plt.savefig('./dist_im/dist.png', bbox_inches='tight')
plt.close()

_plot_fig, _plot_axs = plt.subplots(5, 2, figsize=(10, 24))
_plot_lines = []
for i in range(10):
    mappable = _plot_axs[i % 5, i // 5].pcolormesh(dist0[i, ::-1], vmin=0, vmax=1, cmap='summer')
    #_plot_fig.colorbar(mappable)
    _plot_axs[i % 5, i // 5].set_xticks([])
    _plot_axs[i % 5, i // 5].set_yticks([])
    _plot_axs[i % 5, i // 5].set_xlim([0, dist0.shape[1]])
    _plot_axs[i % 5, i // 5].set_ylim([0, dist0.shape[2]])
plt.savefig('./dist_im/dist0.png', bbox_inches='tight')
plt.close()

_plot_fig, _plot_axs = plt.subplots(5, 2, figsize=(10, 24))
_plot_lines = []
for i in range(10):
    mappable = _plot_axs[i % 5, i // 5].pcolormesh(dist1[i, ::-1], vmin=0, vmax=1, cmap='summer')
    #_plot_fig.colorbar(mappable)
    _plot_axs[i % 5, i // 5].set_xticks([])
    _plot_axs[i % 5, i // 5].set_yticks([])
    _plot_axs[i % 5, i // 5].set_xlim([0, dist1.shape[1]])
    _plot_axs[i % 5, i // 5].set_ylim([0, dist1.shape[2]])
plt.savefig('./dist_im/dist1.png', bbox_inches='tight')
plt.close()
