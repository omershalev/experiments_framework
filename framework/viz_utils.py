import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_2d_trajectory(dfs, labels=None, dir_name=None):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    for df in dfs:
        plt.plot(df['pose.pose.position.x'], df['pose.pose.position.y']) # TODO: change this!!!
    plt.axis('equal')
    if labels is not None:
        plt.legend(labels)
    plt.xlabel('x')
    plt.ylabel('y')
    if dir_name is not None:
        plt.savefig(os.path.join(dir_name, '2d_trajectories'))