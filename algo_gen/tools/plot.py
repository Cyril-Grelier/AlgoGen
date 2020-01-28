# import textwrap
#
# import matplotlib.pyplot as plt
# import numpy as np
import json
from glob import glob
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure

sns.set()


def save_stats(stats, label, name):
    for s in stats:
        s['parameters']['individual'] = 'onemax'
        s['parameters']['termination_condition'] = 'final_condition'
    for i, sl in enumerate(zip(stats, label)):
        s, l = sl
        with open("test_results/" + str(i).zfill(3) + "_" + name + "_" + str(l) + ".json", 'w') as f:
            json.dump(s, f)


def load_stats(name):
    stats = []
    label = []
    files = glob("test_results/*_" + name + "_*.json")
    files.sort()
    for f in files:
        with open(f) as file:
            stats.append(json.load(file))
        label.append(f.split(name + "_")[-1].split(".json")[0])
    return stats, label


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


def show_stats(stats, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    if title:
        plt.title(title)
    plt.xlabel("tours")
    plt.ylabel('fitness')
    plt.xlim(0, len(stats['max_fitness']))
    plt.ylim(-70, 0)  # (-70,0)#(0, 100)  # stats['parameters']['chromosome size'])  # stats['max_fitness'][-1]*1.1

    ax.plot(stats['max_fitness'], color='red', label='max')
    ax.plot(stats['min_fitness'], color='green', label='min')
    ax.plot(stats['mean_fitness'], color='blue', label='mean')
    # ax.plot(stats['fitness_diversity'], color='black', label='fitness_diversity')

    windows_size = 49
    polynomial_order = 3

    # ax.plot(savgol_filter(stats['max_fitness'], windows_size, polynomial_order), color='red', linestyle='dashed',
    #         label='max_fitness soft')
    # ax.plot(savgol_filter(stats['min_fitness'], windows_size, polynomial_order), color='green', linestyle='dashed',
    #         label='min_fitness soft')
    # ax.plot(savgol_filter(stats['mean_fitness'], windows_size, polynomial_order), color='blue', linestyle='dashed',
    #         label='mean_fitness soft')

    plt.legend(title='fitness', loc='lower right')

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


def plot_mean_stats(stats, label, title, diff_labels):
    max_fit = [s['max_fitness'] for s in stats]
    nb_diff_label = len(set(label))
    pack_max_fit = []
    for i in range(nb_diff_label):
        pack_max_fit.append(max_fit[i * 20:(i * 20) + 20])
    figure(figsize=(8, 8))
    for i, m in enumerate(pack_max_fit):
        maxi = 0
        for l in m:
            if len(l) > maxi:
                maxi = len(l)
        for l in m:
            for _ in range(len(l), maxi):
                l.append(l[-1])
        y = list(map(mean, zip(*m)))
        x = list(range(len(y)))
        #         color = ['b','r','g','y','c','m'][i]
        plt.plot(x, y, label=diff_labels[i])  # , color=color)
    plt.title(title)
    if 'ordre 1' in title or 'PMX' in title:
        plt.xlim(0, 10000)
        plt.ylim(-25, 0)
    else:
        plt.xlim(0, 5000)
        plt.ylim(0, 100)
    plt.ylabel("fitness")
    plt.xlabel("tours")
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(title + ".png", pad_inches=0, bbox_inches='tight')


def plot_all_stats(stats, label, title, diff_labels):
    max_fit = [s['max_fitness'] for s in stats]
    # label_bool = [False] * len(diff_labels)
    fig = figure(num=1, figsize=(15, 15))
    for i, y in enumerate(max_fit):
        x = list(range(len(y)))

        # color = ['b', 'r', 'g', 'y', 'c', 'm'][diff_labels.index(label[i])]
        # if not label_bool[diff_labels.index(label[i])]:
        #     label_bool[diff_labels.index(label[i])] = True
        p = plt.plot(x, y, label=label[i])  # , color=color)
        plt.plot(x, y, c=p[0].get_c())
        # else:
        #     plt.plot(x, y, color=color)
    plt.title(title)
    if 'ordre 1' in title or 'PMX' in title:
        plt.xlim(0, 10000)
        plt.ylim(-60, 0)
    else:
        plt.xlim(0, 5000)
        plt.ylim(0, 100)
    # plt.legend()
    ax = fig.gca()  # get the current axis

    # for i, p in enumerate(ax.get_lines()):  # this is the loop to change Labels and colors
    #     if p.get_label() in diff_labels[:i]:  # check for Name already exists
    #         idx = diff_labels.index(p.get_label())  # find ist index
    #         p.set_c(ax.get_lines()[idx].get_c())  # set color
    #         p.set_label('_' + p.get_label())  # hide label in auto-legend
    # plt.legend()
    plt.ylabel("fitness")
    plt.xlabel("tours")
    plt.show()

# 66023c
# b723a5
# ff8d00
# b2e1fd
# cedc00
# f9423a
# c90000
# 6300ff
# 00ffdf
# d2e7ff
# 1fe5b6
# 25f2c4
# b9e7c3
# ffc78e
# f4b3d3
# 7fe5f0
# e763ba
# ff0030
# 00adee
# b4d455
# f42069
# ffcf17
# ffc000
# ffd2d2
# f4ac00
# dbc6ea
# 0c5f2c
# 71a6d2
