import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ML4QS.Python3Code.Chapter4.TemporalAbstraction import NumericalAbstraction
import sys
import os
sys.path.append(os.path.abspath('/Users/andrasjoos/Documents/AI_masters/Period_6/MLQS/MLQS_assignment'))
from ML4QS.Python3Code.Chapter7 import FeatureSelection
import time
from sklearn.preprocessing import LabelEncoder
from src.FrequencyAbstraction import FourierTransformation


pd.options.mode.chained_assignment = None


def calc_aggs(df, cols, window, method="mean"):
    num_abs = NumericalAbstraction()

    for c in cols:
        df[c+" "+method] = num_abs.aggregate_value(df[c], window, method)
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


def feature_selection(train_df, test_df):
    X_train = train_df.drop(columns=['Activity'])
    y_train = train_df['Activity']

    X_test = test_df.drop(columns=['Activity'])
    y_test = test_df['Activity']
    
    feature_sel = FeatureSelection.FeatureSelectionClassification()

    forward_selected_features, ordered_features, ordered_scores = feature_sel.forward_selection(10, X_train, X_test, y_train, y_test)
    #backward_selected_features = feature_sel.backward_selection(8, X_train, y_train)

    forward_df = pd.DataFrame(forward_selected_features, columns=['Forward Selected Features'])
    print("\nForward Selection Results:")
    print(forward_df)

    ordered_df = pd.DataFrame({
        'Ordered Features': ordered_features,
        'Scores': ordered_scores
    })
    ordered_df = ordered_df.sort_values(by='Scores', ascending=False)
    print("\nOrdered Features and Scores:")
    print(ordered_df)

    # backward_df = pd.DataFrame(backward_selected_features, columns=['Backward Selected Features'])
    # print("\nBackward Selection Results:")
    # print(backward_df)

    return forward_selected_features#, backward_selected_features


def split_data(df, split_ratio=0.8, random_state=None):
    # Shuffle the DataFrame
    # df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_shuffled = df

    # Calculate the split index based on the length of the DataFrame
    split_index = int(len(df_shuffled) * split_ratio)

    # Use .iloc for positional slicing
    train_df = df_shuffled.iloc[:split_index]
    test_df = df_shuffled.iloc[split_index:]

    return train_df, test_df


def main():
    print("Feature engineering starts...")
    start_time = time.time()
    df = pd.read_csv('data_agg/point_two_sec_agg_clean.csv')
    df = df.drop(columns=['Datetime_linacc.1'])#, 'Datetime_linacc'])
    df = df.set_index('Time (s)')
    print(df.columns)

    # Perform feature engineering on the training set
    df = aggregation_features(df)
    df = frequency_features(df)
    
    df.to_csv("data_agg/feature_engineered.csv", index=False)

    # # Feature selection
    # print("Feature selection starts...")
    # forward_features = feature_selection(train_df, test_df)
    # sel_features_with_activity = forward_features + ['Activity']
    # train_df = train_df[sel_features_with_activity]
    # test_df = test_df[sel_features_with_activity]
 
    # # Save the feature-engineered training and testing sets
    # train_df.to_csv("data_agg/feature_engineered_train_selected.csv", index=False)
    # test_df.to_csv("data_agg/feature_engineered_test_selected.csv", index=False)

    end_time = time.time()
    print(f"Feature engineering completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
