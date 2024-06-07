#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from src.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection


def remove_start_end(df, secs=5):
    return df[(df["Time (s)"] > secs) & (df["Time (s)"] < max(df["Time (s)"]) - secs)]


def remove_outliers(df, col):
    OutlierDistr = DistanceBasedOutlierDetection()
    df = OutlierDistr.local_outlier_factor(df, [col], "euclidean", 30)
    #print(df)
    return df


for cat in ["Walking", "Sitting", "Cycling", "Sport"]:
    print(cat)
    for f in os.listdir("data/"):
        if cat in f:
            for subf in os.listdir("data/"+f):
                print(f, subf)
                acc = pd.read_csv("data/"+f+"/"+subf+"/Accelerometer.csv")
                loc = pd.read_csv("data/"+f+"/"+subf+"/Location.csv")
                gyro = pd.read_csv("data/"+f+"/"+subf+"/Gyroscope.csv")

                cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)",
                        "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)",
                        "Velocity (m/s)", "Height (m)"]

                acc = remove_start_end(acc)
                loc = remove_start_end(loc)
                gyro = remove_start_end(gyro)

                #remove_outliers(acc)
                #remove_outliers(loc, "Latitude (°)")

                acc.drop_duplicates(subset=cols[:3], inplace=True)

                x = pd.merge_asof(loc, pd.merge_asof(acc, gyro, on="Time (s)", direction="nearest"), on="Time (s)", direction="nearest")

                fig, ax = plt.subplots(3,3)
                fig.set_size_inches(20, 14)
                plt.plot(x['Longitude (°)'], x['Latitude (°)'])
                plt.title("Path")
                for i, c in enumerate(cols):
                    #x[c].plot(ax=ax.flatten()[i])
                    ax.flatten()[i].plot(x["Time (s)"], x[c])
                    ax.flatten()[i].set_title(c)

                plt.suptitle(f+" "+subf)
                plt.show()
                

# %%
