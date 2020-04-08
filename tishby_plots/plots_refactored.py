"""Plot the networks in the information plane"""
import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


def plot_all_epochs(I_XT_array, I_TY_array, epochs_inds, title_str='', save_name='unnamed_plot',
                    x_lim=(0, 5.4), y_lim=(0, 3), fig_size=(14, 10),
                    label_size=34, axis_font=28, bar_font=28, colorbar_axis=(0.925, 0.091, 0.03, 0.8)):
    """Plot the information plane with the epochs in different colors """
    # If we want to plot the train and test error

    f = plt.figure(figsize=fig_size)
    axes = f.add_subplot(111)

    num_info_epochs = len(epochs_inds)
    cmap = plt.get_cmap('gnuplot')

    # For each epoch we have different color
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
