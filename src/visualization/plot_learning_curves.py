import pandas as pd
import matplotlib.pyplot as plt          # plotting
import seaborn as sns
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description='generate f1 learning curve plots')
parser.add_argument('input_folder', required=True, help='f1_metrics')


def RGB_t(r,g,b):
    return str(r/255) + ',' + str(g/255)  + ',' + str(b/255)


def RGB(r,g,b):
    return [(r/255), g/255, b/255]


aspect_bg_color = RGB_t(227, 222, 209)
aspect_orange = RGB(240, 127, 9)
aspect_red = RGB(159, 41, 54)
aspect_blue = RGB(27, 88, 124)
aspect_green = RGB(78, 133, 66)
aspect_purple = RGB(96, 72, 120)
aspect_gray = RGB(50, 50, 50)
aspect_palette = [aspect_orange, aspect_red, aspect_blue, aspect_green, aspect_purple, aspect_gray]


def load_merged_tensorboard_csvs(path):
    return pd.read_csv(path, dtype=np.float32, memory_map=True, index_col='Step')


def plot_data(data_path, plot_path, colors):
    plt.clf()

    data = load_merged_tensorboard_csvs(data_path)
    #     palette = sns.color_palette("mako_r", colors)
    g = sns.lineplot(data=data, palette=aspect_palette[:colors])
    plt.ylim(0, 1.1)
    plt.xlabel('Training Iteration')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show(g)
    #    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


working_dir = '/home/goodwintrr/cebmimic/travis/amia_summit_2019/'
input_file_pattern = os.path.join(working_dir, 'tf_%s_f1_%s.csv')
output_file_pattern = '%ss_%s_f1.pdf'
for ablation, colors in [('encoder', 5), ('cell', 6)]:
    for dataset in ['train', 'devel', 'test']:
        plot_data(input_file_pattern % (ablation, dataset), output_file_pattern % (ablation, dataset), colors)