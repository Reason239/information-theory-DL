"""Plot the networks in the information plane"""
import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import _pickle as cPickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import scipy.stats as sis
import os
import matplotlib.animation as animation
import math
import os.path
import tkinter as tk
from numpy import linalg as LA
from tkinter import filedialog
import sys


# LAYERS_COLORS = ['red', 'blue', 'green', 'yellow', 'pink', 'orange']
#
#
# def get_data(name):
#     """Load data from the given name"""
#     gen_data = {}
#     # new version
#     if os.path.isfile(name + 'data.pickle'):
#         curent_f = open(name + 'data.pickle', 'rb')
#         d2 = cPickle.load(curent_f)
#     # Old version
#     else:
#         curent_f = open(name, 'rb')
#         d1 = cPickle.load(curent_f)
#         data1 = d1[0]
#         data = np.array([data1[:, :, :, :, :, 0], data1[:, :, :, :, :, 1]])
#         # Convert log e to log2
#         normalization_factor = 1 / np.log2(2.718281)
#         epochsInds = np.arange(0, data.shape[4])
#         d2 = {}
#         d2['epochsInds'] = epochsInds
#         d2['information'] = data / normalization_factor
#     return d2
#
#
# def create_color_bar(f, cmap, colorbar_axis, bar_font, epochsInds, title):
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
#     sm._A = []
#     cbar_ax = f.add_axes(colorbar_axis)
#     cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
#     cbar.ax.tick_params(labelsize=bar_font)
#     cbar.set_label(title, size=bar_font)
#     cbar.ax.text(0.5, -0.01, epochsInds[0], transform=cbar.ax.transAxes,
#                  va='top', ha='center', size=bar_font)
#     cbar.ax.text(0.5, 1.0, str(epochsInds[-1]), transform=cbar.ax.transAxes,
#                  va='bottom', ha='center', size=bar_font)
#
#
# def adjustAxes(axes, axis_font=20, title_str='', x_ticks=[], y_ticks=[], x_lim=None, y_lim=None,
#                set_xlabel=True, set_ylabel=True, x_label='', y_label='', set_xlim=True, set_ylim=True, set_ticks=True,
#                label_size=20, set_yscale=False,
#                set_xscale=False, yscale=None, xscale=None, ytick_labels='', genreal_scaling=False):
#     """Organize the axes of the given figure"""
#     if set_xscale:
#         axes.set_xscale(xscale)
#     if set_yscale:
#         axes.set_yscale(yscale)
#     if genreal_scaling:
#         axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#         axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#         axes.xaxis.major.formatter._useMathText = True
#         axes.set_yticklabels(ytick_labels)
#         axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
#     if set_xlim:
#         axes.set_xlim(x_lim)
#     if set_ylim:
#         axes.set_ylim(y_lim)
#     axes.set_title(title_str, fontsize=axis_font + 2)
#     axes.tick_params(axis='y', labelsize=axis_font)
#     axes.tick_params(axis='x', labelsize=axis_font)
#     if set_ticks:
#         axes.set_xticks(x_ticks)
#         axes.set_yticks(y_ticks)
#     if set_xlabel:
#         axes.set_xlabel(x_label, fontsize=label_size)
#     if set_ylabel:
#         axes.set_ylabel(y_label, fontsize=label_size)
#

def plot_all_epochs(I_XT_array, I_TY_array, epochs_inds, title_str='', save_name='unnamed_plot',
                    x_lim=(0, 5.4), y_lim=(0, 3), fig_size=(14, 10),
                    label_size=34, axis_font=28, bar_font=28, colorbar_axis=(0.925, 0.091, 0.03, 0.8)):
    """Plot the information plane with the epochs in different colors """
    # If we want to plot the train and test error

    # axis_font = 28
    # bar_font = 28
    # fig_size = (14, 10)
    # label_size = 34
    # f, (axes) = plt.subplots(1, 1, sharey=True, figsize=fig_size)
    # axes = np.vstack(np.array([axes]))
    # f.subplots_adjust(left=0.097, bottom=0.12, right=.87, top=0.99, wspace=0.03, hspace=0.03)

    # colorbar_axis = [0.905, 0.12, 0.03, 0.82]

    f = plt.figure(figsize=fig_size)
    axes = f.add_subplot(111)

    num_info_epochs = len(epochs_inds)
    cmap = plt.get_cmap('gnuplot')

    # For each epoch we have different color
    # KT ? why + 1
    colors = [cmap(i) for i in np.linspace(0, 1, epochs_inds[-1] + 1)]

    axes.set_title(title_str, fontsize=axis_font + 2)
    axes.tick_params(axis='y', labelsize=axis_font)
    axes.tick_params(axis='x', labelsize=axis_font)
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    x_ticks = list(range(round(x_lim[0]), round(x_lim[1]) + 1))
    y_ticks = list(range(round(y_lim[0]), round(y_lim[1]) + 1))
    axes.set_xticks(x_ticks)
    axes.set_yticks(y_ticks)
    x_label = '$I(X;L + eps)$'
    y_label = '$I(L;Y)$'
    axes.set_xlabel(x_label, fontsize=label_size)
    axes.set_ylabel(y_label, fontsize=label_size)

    # Go over all the epochs and plot then with the right color
    for index_in_range in range(num_info_epochs):
        XT = I_XT_array[index_in_range, :]
        TY = I_TY_array[index_in_range, :]
        axes.plot(XT[:], TY[:], marker='o', linestyle='-', markersize=12, markeredgewidth=0.01,
                  linewidth=0.2, color=colors[int(epochs_inds[index_in_range])])

    # Save the figure and add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar_ax = f.add_axes(colorbar_axis)
    cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=bar_font)
    cbar.set_label('Epochs', size=bar_font)
    cbar.ax.text(0.5, -0.01, epochs_inds[0], transform=cbar.ax.transAxes,
                 va='top', ha='center', size=bar_font)
    cbar.ax.text(0.5, 1.0, epochs_inds[-1], transform=cbar.ax.transAxes,
                 va='bottom', ha='center', size=bar_font)

    f.savefig(save_name + '.jpg', dpi=500, format='jpg')

#
# def plot_figures(str_names, mode, save_name):
#     """Plot the data in the given names with the given mode"""
#
#     # one figure
#     axis_font = 28
#     bar_font = 28
#     fig_size = (14, 10)
#     font_size = 34
#     f, (axes) = plt.subplots(1, 1, sharey=True, figsize=fig_size)
#     axes = np.vstack(np.array([axes]))
#     f.subplots_adjust(left=0.097, bottom=0.12, right=.87, top=0.99, wspace=0.03, hspace=0.03)
#     colorbar_axis = [0.905, 0.12, 0.03, 0.82]
#     xticks = [1, 3, 5, 7, 9, 11]
#     yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
#
#     sizes = [[-1]]
#     title_strs = [['', '']]
#
#     # Go over all the files
#     for i in range(len(str_names)):
#         for j in range(len(str_names[i])):
#             name_s = str_names[i][j]
#             # Load data for the given file
#             data_array = utils.get_data(name_s)
#             data = np.squeeze(np.array(data_array['information']))
#             I_XT_array = np.array(extract_array(data, 'local_IXT'))
#             I_TY_array = np.array(extract_array(data, 'local_ITY'))
#             epochsInds = data_array['params']['epochsInds']
#             # Plot it
#             plot_all_epochs(data_array, I_XT_array, I_TY_array, axes, epochsInds, f, i, j, sizes[i][j], font_size,
#                             yticks, xticks,
#                             colorbar_axis, title_strs[i][j], axis_font, bar_font, save_name)
#     plt.show()
