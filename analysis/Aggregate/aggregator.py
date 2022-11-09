import json
import csv
import sys
import os
import glob
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

# Author: Andrew Jo
# Run:
# python aggregator.py Normalize(or Aggregate)
#
# Output: Save heatmap of all the csv files in the provided folder name

if(len(sys.argv) == 2) and (sys.argv[1] == 'Aggregate' or sys.argv[1] == 'Normalize' or sys.argv[1] == 'Diff' or sys.argv[1] == 'AggregateRepetitionsRemoved'):
    folder = sys.argv[1]

    path = os.path.dirname(os.path.abspath(__file__)) + '/' + folder
    filenames = glob.glob(path + "/*.csv")

    for i in range(len(filenames)):

        fileName = os.path.basename(filenames[i])

        data = pd.read_csv(filenames[i], delimiter=',', index_col=0)
        print data.shape

        ticks = []
        for r in data:
            ticks.append(r)

        plt.figure(figsize=(20,15))
        g = sns.heatmap(data, cmap="Greens", linewidth=0, xticklabels=ticks,yticklabels=ticks)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 5)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 5)
        g.xaxis.tick_top()
        g.tick_params(axis='both', which='major', pad=5)
        plt.savefig('../Aggregate/Heatmap/' + folder + '/' + fileName + '.png')
else:
    print("Double check the input parameters above :)")