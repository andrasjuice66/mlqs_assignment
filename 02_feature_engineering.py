import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ML4QS.Python3Code.Chapter4.TemporalAbstraction import NumericalAbstraction
from sklearn.preprocessing import LabelEncoder
import time


def aggregation_features(df):

    # Initialize the NumericalAbstraction class
    num_abs = NumericalAbstraction()

    # List of columns to apply the aggregation functions
    cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)",
                        "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)",
                        "Velocity (m/s)", "Height (m)", "Linear Acceleration x (m/s^2)", 
                        "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]
    window_size = 10

    df[cols] = df[cols].abs()

    # Apply various aggregations
    aggregations = ['mean', 'max', 'min', 'std', 'slope']
    for agg in aggregations:
        df = num_abs.abstract_numerical(df, cols, window_size, agg)
    return df

def frequency_features(df):
    return df

def main():
    print("Feature enigneering starts...")
    start_time = time.time()
    df = pd.read_csv('data_agg/final_aggregated_output_no_outlier.csv', index_col=0)

    df = aggregation_features(df)
    end_time_agg = time.time()
    print(f"Aggregation feature engineering ended in  {end_time_agg - start_time:.2f} seconds.")

    df = frequency_features(df)
    end_time_freq = time.time()
    print(f"Frequency feature engineering ended in {end_time_agg - end_time_freq:.2f} seconds.")

    df.to_csv(f"data_agg/feature_engineered.csv", index=False)
    end_time = time.time()
    print(f"Exported feature_engineered.csv in  {end_time_freq - end_time:.2f} seconds.")
    print(f"Feature engineering endid in total of {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()