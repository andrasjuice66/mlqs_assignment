import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import time
from src.OutlierDetection import DistanceBasedOutlierDetection
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util.plot_util import plot_lof_2d_grad, plot_lof_2d_circle, plot_lof_3d_grad ,plot_lof_3d_circle


def remove_start_end(df, secs=5):
    return df[(df["Time (s)"] > secs) & (df["Time (s)"] < max(df["Time (s)"]) - secs)]

def calculate_lof(df, cols):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    #X = df[cols].values
    y_pred = lof.fit_predict(df[cols])
    
    df['lof'] = y_pred
    df['lof_factor'] = -lof.negative_outlier_factor_  # Negate to align with typical plotting conventions
    return df

def outlier_search(df, cols, method='lof'):
    for col in cols:
        if method == 'lof':
            df = calculate_lof(df, cols)
        elif method == 'zscore':
            z_scores = stats.zscore(df[col])
            df.loc[abs(z_scores) > 3, col] = np.nan
        elif method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), col] = np.nan
    if method == 'lof':
        df.loc[df['lof'] == -1, cols] = np.nan
    df[cols].interpolate(method='linear', inplace=True)
    return df


if __name__ == "__main__":
    for cat in ["Walking", "Sitting", "Cycling", "Sport"]:
        print(cat)
        for f in os.listdir("data_1/"):
            if cat in f:
                for subf in os.listdir("data_1/" + f):
                    start_time = time.time()
                    print(f, subf)
                    acc = pd.read_csv("data_1/" + f + "/" + subf + "/Accelerometer.csv")
                    loc = pd.read_csv("data_1/" + f + "/" + subf + "/Location.csv")
                    gyro = pd.read_csv("data_1/" + f + "/" + subf + "/Gyroscope.csv")

                    cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)",
                            "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)",
                            "Velocity (m/s)", "Height (m)"]

                    acc = remove_start_end(acc)
                    loc = remove_start_end(loc)
                    gyro = remove_start_end(gyro)

                    
                    acc_cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)", "Time (s)"]
                    loc_cols = ["Latitude (°)", "Longitude (°)", "Time (s)"]
                    loc_height_col = ["Latitude (°)", "Longitude (°)", "Height (m)", "Time (s)"]
                    gyro_cols = ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)', "Time (s)"]
                    velocity_col = ["Velocity (m/s)"]

                    # Detect and remove outliers for accelerometer data
                    acc = calculate_lof(acc, acc_cols)
                    plot_lof_3d_grad(acc, acc_cols, cat)

                    # Detect and remove outliers for gyroscope data
                    gyro = calculate_lof(gyro, gyro_cols)
                    plot_lof_3d_grad(gyro, gyro_cols, cat)

                    # Detect and remove outliers for loc data
                    loc = calculate_lof(loc, loc_cols)
                    plot_lof_2d_grad(loc, loc_cols, cat)

                    # Detect and remove outliers for loc_height data
                    # loc = calculate_lof(loc,  loc_height_col)
                    # plot_lof_3d_grad(loc, loc_height_col, cat)


                    # Detect and remove outliers for location data
                    #loc = outlier_search(loc, loc_cols, method='lof')
                    #loc = outlier_search(loc, loc_height_col, method='zscore')

                    # Detect and remove outliers for velocity data
                    #loc = outlier_search(loc, velocity_col, method='iqr')

                    # # Merge dataframes
                    # df = pd.merge_asof(loc, pd.merge_asof(acc, gyro, on="Time (s)", direction="nearest"), on="Time (s)", direction="nearest")



                    # # Plotting with outliers
                    # total_outliers = plot_with_outliers(df, cols)
                    # end_time = time.time()
                    # processing_time = end_time - start_time

                    # print(f"Processing time for {subf}: {processing_time:.2f} seconds")
                    # print(f"Total outliers in {subf}: {total_outliers}")
