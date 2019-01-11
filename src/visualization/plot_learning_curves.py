import pandas as pd
import matplotlib.pyplot as plt          # plotting
import seaborn as sns                    # plotting presets
import numpy as np

import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='generate f1 learning curve plots')
parser.add_argument('input_folder', help='f1_metrics')


def rgb_str(r, g, b):
    '''Convert RGB values into a string'''
    return str(r/255) + ',' + str(g/255)  + ',' + str(b/255)


def rgb(r, g, b):
    '''Convert RGB values into a list'''
    return [(r/255), g/255, b/255]


# Use the "Aspect" colorscheme from MS Office so these plots match my other figures
# Custom background color
aspect_bg_color = rgb_str(227, 222, 209)
# Color palatte
aspect_orange = rgb(240, 127, 9)
aspect_red = rgb(159, 41, 54)
aspect_blue = rgb(27, 88, 124)
aspect_green = rgb(78, 133, 66)
aspect_purple = rgb(96, 72, 120)
aspect_gray = rgb(50, 50, 50)
aspect_palette = [aspect_orange, aspect_red, aspect_blue, aspect_green, aspect_purple, aspect_gray]


def load_merged_tensorboard_csvs(path):
    '''Load csv file into a pandas dataframe'''
    return pd.read_csv(path, dtype=np.float32, memory_map=True, index_col='Step')


def plot_data(data_path, plot_path, colors):
    # Clear any previous figure just incase
    plt.clf()

    # Load data from CSV
    data = load_merged_tensorboard_csvs(data_path)

    # Plot our data on a line
    g = sns.lineplot(data=data, palette=aspect_palette[:colors])
    # Set our y-axis to be between 0 and 1.1
    plt.ylim(0, 1.1)
    # Set the label for our x-axis
    plt.xlabel('Training Iteration')
    # Set the legend to appear in the lower-right of the image
    plt.legend(loc='lower right')
    # Reduce space around image
    plt.tight_layout()
    # Save the figure to the given path, using a tight bounding box (again reduce space around image)
    plt.savefig(plot_path, bbox_inches='tight')
    # Clear the figure
    plt.clf()


def main():
    args = parser.parse_args()
    # Input files are CSV files of the form tf_[encoder/cell]_f1_[train/deve/test].csv
    input_file_pattern = os.path.join(args.input_folder, 'tf_%s_f1_%s.csv')
    output_file_pattern = '%ss_%s_f1.pdf'
    # There were 5 encodes and 6 cells, so we need to give the number of colors
    for ablation, colors in [('encoder', 5), ('cell', 6)]:
        for dataset in ['train', 'devel', 'test']:
            plot_data(input_file_pattern % (ablation, dataset), output_file_pattern % (ablation, dataset), colors)


if __name__ == "__main__":
    main()
