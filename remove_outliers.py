#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from src.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection


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

                    acc_cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]
                    loc_cols = ["Latitude (°)", "Longitude (°)"]
                    loc_height_col = ["Height (m)"]
                    gyro_cols = ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)']
                    velocity_col = ["Velocity (m/s)"]

                    # Detect and remove outliers for accelerometer data
                    acc = outlier_search(acc, acc_cols, method='lof')

                    # Detect and remove outliers for gyroscope data
                    gyro = outlier_search(gyro, gyro_cols, method='lof')

                    # Detect and remove outliers for location data
                    loc = outlier_search(loc, loc_cols, method='lof')
                    loc = outlier_search(loc, loc_height_col, method='zscore')

                    # Detect and remove outliers for velocity data
                    loc = remove_outliers(loc, velocity_col, method='iqr')
                    
                    # Merge dataframes
                    x = pd.merge_asof(loc, pd.merge_asof(acc, gyro, on="Time (s)", direction="nearest"), on="Time (s)", direction="nearest")

                    # Plotting
                    fig, ax = plt.subplots(3, 3)
                    fig.set_size_inches(20, 14)

                    ax[0, 0].plot(x['Longitude (°)'], x['Latitude (°)'])
                    ax[0, 0].set_title("Path")

                    for i, c in enumerate(cols):
                        ax.flatten()[i + 1].plot(x["Time (s)"], x[c])
                        ax.flatten()[i + 1].set_title(c)
                        if 'lof' in x.columns:
                            outlier_indices = x[x['lof'] == -1].index
                            ax.flatten()[i + 1].scatter(x.loc[outlier_indices, "Time (s)"], x.loc[outlier_indices, c], color='red')

                    plt.suptitle(f"{f} {subf}")
                    plt.show()
