import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

import config

def plot_2d_trajectory(dfs, labels=None, colors=None, file_name=None, show=False, xlim=None, ylim=None, label_x_axis=True, label_y_axis=True):
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['legend.loc'] = 'best'
    for idx, df in enumerate(dfs):
        if colors is not None:
            plt.plot(df[df.columns[0]], df[df.columns[1]], colors[idx])
        else:
            plt.plot(df[df.columns[0]], df[df.columns[1]])
    if labels is not None:
        plt.legend(labels)
    if label_x_axis:
        plt.xlabel('x [m]')
    else:
        plt.tick_params(labelbottom=False)
    if label_y_axis:
        plt.ylabel('y [m]')
    else:
        plt.tick_params(labelleft=False)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is None and ylim is None:
        plt.axis('equal')
    if file_name is not None:
        plt.savefig(os.path.join(file_name))
    if show:
        plt.show()


def plot_line_with_sleeve(vector, sleeve_width, color, dashed=False):
    upper_bound = vector + sleeve_width / 2
    lower_bound = vector - sleeve_width / 2
    if dashed:
        plt.plot(vector, color, linestyle='--')
    else:
        plt.plot(vector, color)
    plt.fill_between(vector.index, upper_bound, lower_bound, facecolor=color, alpha=0.2)


def show_image(window_name, image, wait_key=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.screen_resolution[0], config.screen_resolution[1])
    cv2.imshow(window_name, image)
    if wait_key:
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)