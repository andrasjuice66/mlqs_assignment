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
    # Instantiate the FourierTransformation class
    ft = FourierTransformation()

    # Define the columns to transform, window size, and sampling rate
    columns = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']
    window_size = 5
    sampling_rate = 5  

    df_transformed = ft.abstract_frequency(df, columns, window_size, sampling_rate)
    return df_transformed

def split_data(df):
    # Define the split index
    split_index = int(0.8 * len(df))
    
    # Split the data into training and testing sets
    train_df = df[:split_index]
    test_df = df[split_index:]
    
    return train_df, test_df

def main():
    print("Feature engineering starts...")
    start_time = time.time()
    df = pd.read_csv('data_agg/point_two_sec_agg_clean.csv', index_col=0)

    # Split the data into training and testing sets
    train_df, test_df = split_data(df)

    # Perform feature engineering on the training set
    train_df = aggregation_features(train_df)
    train_df = frequency_features(train_df)

    # Perform feature engineering on the testing set
    test_df = aggregation_features(test_df)
    test_df = frequency_features(test_df)

    # Save the feature-engineered training and testing sets
    train_df.to_csv("data_agg/feature_engineered_train.csv", index=False)
    test_df.to_csv("data_agg/feature_engineered_test.csv", index=False)

    end_time = time.time()
    print(f"Feature engineering completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
