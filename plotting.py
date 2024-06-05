import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats


def remove_start_end(df, secs=5):
    return df[(df["Time (s)"] > secs) & (df["Time (s)"] < max(df["Time (s)"]) - secs)]


def remove_outliers(df):
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


for cat in ["Walking", "Sitting", "Cycling"]:
    print(cat)
    for f in os.listdir("data/"):
        if cat in f:
            for subf in os.listdir("data/"+f):
                print(f, subf)
                acc = pd.read_csv("data/"+f+"/"+subf+"/Accelerometer.csv")
                loc = pd.read_csv("data/"+f+"/"+subf+"/Location.csv")

                acc = remove_start_end(acc)
                loc = remove_start_end(loc)

                remove_outliers(acc)
                remove_outliers(loc)

                x = pd.merge_asof(loc, acc, on="Time (s)", direction="nearest", tolerance=0.1)

                fig, ax = plt.subplots(2,3)
                fig.set_size_inches(20, 8)
                plt.plot(x['Longitude (°)'], x['Latitude (°)'])
                plt.title("Path")
                for i, c in enumerate(x.columns[3:5].append(x.columns[8:])):
                    x[c].plot(ax=ax.flatten()[i])
                    ax.flatten()[i].set_title(c)

                plt.suptitle(f+" "+subf)
                plt.show()
                
# %%
