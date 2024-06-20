import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import time
from src.OutlierDetection import DistanceBasedOutlierDetection
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util.plot_util import plot_lof_2d_grad, plot_lof_2d_circle, plot_lof_3d_grad ,plot_lof_3d_circle


def remove_start_end(df, secs=5):
    return df[(df["Time (s)"] > secs) & (df["Time (s)"] < max(df["Time (s)"]) - secs)]


def calculate_lof(df, cols, label):
    if not all(col in df.columns for col in cols):
        missing_cols = [col for col in cols if col not in df.columns]
        #print(f"Missing columns for LOF calculation: {missing_cols}")
        return df
    
    df[cols] = df[cols].apply(lambda x: x.fillna(x.median()))
    #df[cols] = df[cols].interpolate()

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    y_pred = lof.fit_predict(df[cols])
    df[f'lof_{label}'] = y_pred
    df[f'lof_factor_{label}'] = -lof.negative_outlier_factor_  # Negate to align with typical plotting conventions

    return df


def remove_outliers_and_impute(df, cols, label):
    df.loc[df[f'lof_{label}'] == -1, cols] = pd.NA

    # Forward fill NaNs
    df[cols] = df[cols].ffill()
    df.drop(columns = [f'lof_{label}', f'lof_factor_{label}'])

    return df


def correct_column_names(df, corrections):
    return df.rename(columns=corrections)


def add_missing_datetime_column(folder_path):
    # Extract start time from the folder name
    folder_name = os.path.basename(folder_path)
    
    try:
        start_time_str = ' '.join(folder_name.split()[-2:])
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H-%M-%S")
    except ValueError as e:
        #print(f"Skipping folder {folder_path} due to ValueError: {e}")
        return

    # Iterate over each CSV file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip non-CSV files
        if not filename.endswith('.csv'):
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Drop rows where 'Time (s)' is missing
            df = drop_rows_with_no_time(df)
            
            # Ensure the Time (s) column exists and the DataFrame is not empty
            if 'Time (s)' in df.columns and not df.empty:
                df['Datetime'] = df['Time (s)'].apply(lambda x: start_time + timedelta(seconds=x))
                df.to_csv(file_path, index=False)
                #print(f"Updated {filename} with Datetime column.")
            else:
                pass
                # print(f"Skipped {filename}: 'Time (s)' column not found or DataFrame is empty.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def drop_rows_with_no_time(df):
    """
    Drop rows where there is no value in the 'Time (s)' column.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame with rows removed where 'Time (s)' is NaN or missing.
    """
    if 'Time (s)' in df.columns:
        #print(df.head)
        # Drop rows where 'Time (s)' is NaN or an empty string
        df_cleaned = df[df['Time (s)'].notna() & (df['Time (s)'] != '')]
        #print(f"Dropped rows: {len(df) - len(df_cleaned)}")
        return df_cleaned
    else:
        #print("The column 'Time (s)' does not exist in the DataFrame.")
        return df
    

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


def concatenate_single_experiments():
    for cat in ["Walking", "Sitting", "Cycling", "Sport"]:
        for f in os.listdir("data/"):
            if cat in f:
                for subf in os.listdir("data/" + f):

                    add_missing_datetime_column("data/" + f + "/" + subf)
                    start_time = time.time()
                    subf_path = os.path.join("data", f, subf)
                    if not os.path.isdir(subf_path):
                        continue  # Skip if it's not a directory
                    
                    acc = pd.read_csv("data/" + f + "/" + subf + "/Accelerometer.csv")
                    acc = correct_column_names(acc, column_corrections["Accelerometer"])

                    gyro = pd.read_csv("data/" + f + "/" + subf + "/Gyroscope.csv")
                    gyro = correct_column_names(gyro, column_corrections["Gyroscope"])

                    linacc_path_1 = "data/" + f + "/" + subf + "/Linear Acceleration.csv"
                    linacc_path_2 = "data/" + f + "/" + subf + "/Linear Accelerometer.csv"

                    if os.path.exists(linacc_path_1):
                        linacc = pd.read_csv(linacc_path_1)
                    elif os.path.exists(linacc_path_2):
                        linacc = pd.read_csv(linacc_path_2)
                    else:
                        #print(f"Warning: Neither Linear Acceleration.csv nor Linear Accelerometer.csv found in {subf_path}.")
                        linacc = pd.DataFrame()  
                    linacc = correct_column_names(linacc, column_corrections["Linear Acceleration"])

                    loc = pd.read_csv("data/" + f + "/" + subf + "/Location.csv")
                    loc = correct_column_names(loc, column_corrections["Location"])

                    cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)",
                            "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)",
                            "Velocity (m/s)", "Height (m)", "Linear Acceleration x (m/s^2)", 
                            "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]
                    
                    acc = remove_start_end(acc)
                    loc = remove_start_end(loc)
                    gyro = remove_start_end(gyro)
                    linacc = remove_start_end(linacc)

                    acc = acc.rename(columns={"Datetime": "Datetime_acc"})
                    loc = loc.rename(columns={"Datetime": "Datetime_loc"})
                    gyro = gyro.rename(columns={"Datetime": "Datetime_gyro"})
                    linacc = linacc.rename(columns={"Datetime": "Datetime_linacc"})

                    # print(df[["Latitude (°)", "Longitude (°)"]].isna().sum())

                    loc['Displacement (m)'] = haversine(loc['Latitude (°)'].iloc[0], loc['Longitude (°)'].iloc[0], loc['Latitude (°)'].iloc[-1], loc['Longitude (°)'].iloc[-1])
                    try:
                        loc['Distance From Last (m)'] = haversine(loc['Latitude (°)'].shift().astype(float), loc['Longitude (°)'].shift().astype(float), loc.loc[1:, 'Latitude (°)'].astype(float), loc.loc[1:, 'Longitude (°)'].astype(float))
                    except:
                        loc['Distance From Last (m)'] = 0

                    loc["Distance From Last (m)"] = loc["Distance From Last (m)"].fillna(0)
            
                    df = pd.merge_asof(linacc, pd.merge_asof(loc, pd.merge_asof(acc, gyro, on="Time (s)", direction="nearest"), on="Time (s)", direction="nearest"), on="Time (s)", direction="nearest")
                    
                    #Activity column
                    df['Activity'] = cat

                    #Remove rows with no Times (s)
                    df = drop_rows_with_no_time(df)

                    # Calculate LOF
                    df = calculate_lof(df, acc_cols, 'acc')
                    df = calculate_lof(df, gyro_cols, 'gyro')
                    df = calculate_lof(df, loc_cols, 'loc')
                    df = calculate_lof(df, linacc_cols, 'linacc')

                    # Create the output directory if it doesn't exist
                    output_dir = "data_processed/" + f + "/" + subf
                    os.makedirs(output_dir, exist_ok=True)

                    df.drop_duplicates(inplace=True)

                    # Write out the merged dataframe to CSV
                    df.to_csv(output_dir + f"/{subf}.csv", index=False)

                    end_time = time.time()
                    print(f"Processed {subf} in {f} for category {cat} in {end_time - start_time:.2f} seconds.")


def concatenate_and_sort_nested_csv_files(base_directory, sort_column):
    df_list = []
    
    # Traverse the base directory and read CSV files
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.csv'):
                
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df_list.append(df)
    
    # Concatenate all DataFrames and sort by the specified column
    concatenated_df = pd.concat(df_list, ignore_index=True)
    sorted_df = concatenated_df.sort_values(by=sort_column)
    return sorted_df


# Define a custom resampling function to handle non-numeric 'Activity' column
def resample_with_activity(data, interval):
    numeric_df = data.drop(columns=['Activity', 'Datetime_linacc']).resample(interval).mean()
    activity_df = data['Activity'].resample(interval).first()
    datetime_df = data['Datetime_linacc'].resample(interval).first()
    return numeric_df.join(activity_df).join(datetime_df)


column_corrections = {
        "Accelerometer": {
                "X (m/s^2)": "Acceleration x (m/s^2)",
                "Y (m/s^2)": "Acceleration y (m/s^2)",
                "Z (m/s^2)": "Acceleration z (m/s^2)"
        },
        "Gyroscope": {
                "X (rad/s)": "Gyroscope x (rad/s)",
                "Y (rad/s)": "Gyroscope y (rad/s)",
                "Z (rad/s)": "Gyroscope z (rad/s)"
        },
        "Linear Acceleration": {
                "X (m/s^2)": "Linear Acceleration x (m/s^2)",
                "Y (m/s^2)": "Linear Acceleration y (m/s^2)",
                "Z (m/s^2)": "Linear Acceleration z (m/s^2)"},
         "Location": {
                "Vertical Accuracy (°)": "Vertical Accuracy (m)"}}


acc_cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]
loc_height_col = ["Latitude (°)", "Longitude (°)", "Height (m)"]
gyro_cols = ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)']
loc_cols = ["Latitude (°)", "Longitude (°)", "Horizontal Accuracy (m)", "Vertical Accuracy (m)"]#, "Height (m)", "Velocity (m/s), "Direction (°)","]
linacc_cols = [ "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)", "Linear Acceleration x (m/s^2)"]


def main():
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # Concatenate single experiments
    start = time.time()
    concatenate_single_experiments()
    end = time.time()
    print(f"Time taken for concatenating single experiments: {end - start:.2f} seconds")

    # Concatenate and sort the CSV files by 'Datetime_linacc'
    start = time.time()
    sorted_nested_df = concatenate_and_sort_nested_csv_files("data_processed", 'Datetime_linacc')
    #sorted_nested_output_file = 'data_agg/final_aggregated_output.csv'
    #sorted_nested_df.to_csv(sorted_nested_output_file, index=False)
    end = time.time()
    print(f"Time taken for concatenating and sorting nested CSV files: {end - start:.2f} seconds")

    # Drop NaN and unnecessary columns
    start = time.time()
    df = sorted_nested_df
    df = df.dropna(subset=["Time (s)"])
    df.drop(columns=["Direction (°)"], inplace=True)
    df["Height (m)"].fillna(0, inplace=True)
    df["Velocity (m/s)"].fillna(0, inplace=True)
    end = time.time()
    print(f"Time taken for dropping NaN and unnecessary columns: {end - start:.2f} seconds")

    # Remove outliers and impute missing values
    start = time.time()
    df = remove_outliers_and_impute(df, acc_cols, 'acc')
    df = remove_outliers_and_impute(df, gyro_cols, 'gyro')
    df = remove_outliers_and_impute(df, loc_cols, 'loc')
    df = remove_outliers_and_impute(df, linacc_cols, 'linacc')
    end = time.time()
    print(f"Time taken for removing outliers and imputing missing values: {end - start:.2f} seconds")

    # Drop duplicates and unnecessary columns
    start = time.time()
    df.drop_duplicates(inplace=True)
    df.drop(columns=df.columns[-8:], inplace=True)
    df.set_index(pd.to_datetime(df['Datetime_linacc']), inplace=True)
    df.drop(columns=['Datetime_acc', 'Datetime_loc', 'Datetime_gyro'], inplace=True)
    df.drop(columns=["Latitude (°)", "Longitude (°)", "Horizontal Accuracy (m)", "Vertical Accuracy (m)"], inplace=True)
    df.to_csv(f"data_agg/final_aggregated_output_no_outlier.csv")
    end = time.time()
    print(f"Time taken for dropping duplicates and unnecessary columns: {end - start:.2f} seconds")

    # Resample the data into 5-second and 0.2-second intervals
    start = time.time()
    df_5s = resample_with_activity(df, '5S')
    df_0_2s = resample_with_activity(df, '200ms')
    df_5s_clean = df_5s.dropna()
    df_0_2s_clean = df_0_2s.dropna()
    df_5s_clean.to_csv("data_agg/five_sec_agg_clean.csv")
    df_0_2s_clean.to_csv("data_agg/point_two_sec_agg_clean.csv")
    end = time.time()
    print(f"Time taken for resampling and saving cleaned data: {end - start:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
