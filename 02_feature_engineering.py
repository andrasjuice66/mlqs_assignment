import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ML4QS.Python3Code.Chapter4.TemporalAbstraction import NumericalAbstraction
from sklearn.preprocessing import LabelEncoder
from src.FrequencyAbstraction import FourierTransformation
import time


def calc_aggs(df, cols, window, method="mean"):
    num_abs = NumericalAbstraction()

    for c in cols:
        df[c+""+method] = num_abs.aggregate_value(df[c], window, method)
    return df


def aggregation_features(df, window_size=10):

    # Initialize the NumericalAbstraction class

    # List of columns to apply the aggregation functions
    cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)",
            "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)",
            "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)",
            "Velocity (m/s)", "Height (m)"]

    # df[cols] = df[cols].abs()

    # Apply various aggregations
    aggregations = ['mean', 'max', 'min', 'std', 'slope']
    for agg in aggregations:
        print("Applying method", agg)
        df = calc_aggs(df, cols, window_size, method=agg)
    return df


def frequency_features(df):
    # Convert the 'Datetime_linacc' column to datetime and set as index
    # df['Datetime_linacc'] = pd.to_datetime(df['Datetime_linacc'])
    # df.set_index('Datetime_linacc', inplace=True)

    # Instantiate the FourierTransformation class
    ft = FourierTransformation()

    # Define the columns to transform, window size, and sampling rate
    columns = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']
    window_size = 5
    sampling_rate = 5  

    df_transformed = ft.abstract_frequency(df, columns, window_size, sampling_rate)
    return df_transformed


def main():
    print("Feature enigneering starts...")
    start_time = time.time()
    df = pd.read_csv('data_agg/final_aggregated_output_no_outlier.csv', index_col=0)

    df = aggregation_features(df)
    end_time_agg = time.time()
    print(f"Aggregation feature engineering ended in {end_time_agg - start_time:.2f} seconds.")

    df = frequency_features(df)
    end_time_freq = time.time()
    print(f"Frequency feature engineering ended in {end_time_agg - end_time_freq:.2f} seconds.")

    df.to_csv(f"data_agg/feature_engineered.csv", index=False)
    end_time = time.time()
    print(f"Exported feature_engineered.csv in {end_time_freq - end_time:.2f} seconds.")
    print(f"Feature engineering endid in total of {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
