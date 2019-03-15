import glob
import re

import pandas as pd
import matplotlib.pyplot as plt          # plotting
import seaborn as sns
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='generate learning curve plots')
parser.add_argument('--pattern', metavar='PATTERN', required=True, help='pattern to locate series names from data filenames')
parser.add_argument('--input', metavar='FILE', nargs='+', help='csv files containing data for each series')
parser.add_argument('--output', metavar='FILE', required=True, help='output file')


def rgb_str(r, g, b):
    return str(r/255) + ',' + str(g/255) + ',' + str(b/255)


def rgb(r, g, b):
    return [(r/255), g/255, b/255]


aspect_bg_color = rgb_str(227, 222, 209)
aspect_orange = rgb(240, 127, 9)
aspect_red = rgb(159, 41, 54)
aspect_blue = rgb(27, 88, 124)
aspect_green = rgb(78, 133, 66)
aspect_purple = rgb(96, 72, 120)
aspect_gray = rgb(50, 50, 50)
aspect_palette = [aspect_orange, aspect_red, aspect_blue, aspect_green, aspect_purple, aspect_gray]


def main(args):
    series_pattern = re.compile(args.pattern)
    print('Compiled series regex:', series_pattern)
    curves = []
    inputs = [filename for input_ in args.input for filename in glob.glob(input_)]
    print('Reading input files:', inputs)
    for curve_csv in inputs:
        match = series_pattern.search(curve_csv)
        series_name = match.group(1)
        curve = pd.read_csv(curve_csv, dtype=np.float32, memory_map=True, index_col='Step', header=0, names=['Step', series_name], usecols=[1, 2])
        curves.append(curve)

    data = curves[0]
    for curve in curves[1:]:
        data = data.join(curve)

    plt.clf()
    #     palette = sns.color_palette("mako_r", colors)
    sns.set_style(style={
        "axes.grid": True,
        "axes.facecolor": aspect_bg_color,
        "axes.edgecolor": "white",
        "grid.color": "white",

        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
    })
    g = sns.lineplot(data=data, palette=aspect_palette[:len(curves)], linewidth=2.5)
    plt.ylim(0.0, 1.09)
    plt.xlabel('Training Iteration')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
