import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

import config

def plot_2d_trajectory(dfs, labels=None, file_name=None, show=False, xlim=None, ylim=None):
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['legend.loc'] = 'best'
    fig = plt.figure()
    for df in dfs:
        plt.plot(df[df.columns[0]], df[df.columns[1]])
    if labels is not None:
        plt.legend(labels)
    plt.xlabel('x')
    plt.ylabel('y')
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


def show_image(window_name, image, wait_key=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.screen_resolution[0], config.screen_resolution[1])
    cv2.imshow(window_name, image)
    if wait_key:
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)