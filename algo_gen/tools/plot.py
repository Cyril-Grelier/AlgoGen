import textwrap

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = f"x={xmax}, y={ymax}"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def show_stats(stats):
    fig, ax = plt.subplots(figsize=(10, 10))

    # plt.xlim(0, len(stats['max_fitness']))
    # plt.ylim(0, 100)  # stats['parameters']['chromosome size'])  # stats['max_fitness'][-1]*1.1

    ax.plot(stats['max_fitness'], color='red', label='max_fitness')
    ax.plot(stats['min_fitness'], color='green', label='min_fitness')
    ax.plot(stats['mean_fitness'], color='blue', label='mean_fitness')
    # ax.plot(stats['fitness_diversity'], color='black', label='fitness_diversity')

    windows_size = 49
    polynomial_order = 3

    # ax.plot(savgol_filter(stats['max_fitness'], windows_size, polynomial_order), color='red', linestyle='dashed',
    #         label='max_fitness soft')
    # ax.plot(savgol_filter(stats['min_fitness'], windows_size, polynomial_order), color='green', linestyle='dashed',
    #         label='min_fitness soft')
    # ax.plot(savgol_filter(stats['mean_fitness'], windows_size, polynomial_order), color='blue', linestyle='dashed',
    #         label='mean_fitness soft')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plt.title("\n".join(textwrap.wrap(str(stats['parameters']['mutation'][1]), 120)))
    plt.show()

    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    # ax.plot(stats['diversity'], color='yellow', label='diversity')
    # ax.plot(stats['max_age'], color='c', label='max_age')
    # ax.plot(stats['mean_age'], color='m', label='mean_age')
    #
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    # ax.plot(stats['total_fitness'], color='black', label='total_fitness')
    #
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #
    # plt.show()
