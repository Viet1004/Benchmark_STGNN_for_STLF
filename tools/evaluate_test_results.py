import os
import pandas as pd
import numpy as np
import argparse
import re
from glob import glob
from pathlib import Path

def report_household_level(y, y_hat):
    mae = np.mean(np.abs(y - y_hat))
    mape = np.mean(np.abs((y - y_hat) / y))
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    return mae, mape, rmse

def report_aggregate_level(y, y_hat):
    y_agg = np.sum(y, axis=1)
    y_hat_agg = np.sum(y_hat, axis=1)
    mae = np.mean(np.abs(y_agg - y_hat_agg))
    mape = np.mean(np.abs((y_agg - y_hat_agg) / y_agg)) 
    rmse = np.sqrt(np.mean((y_agg - y_hat_agg) ** 2))
    return mae, mape, rmse

def process_folder(folder_path, df, df_agg):
    files = glob(os.path.join(folder_path, "*.npy"))
    y = None
    for file in files:
        if "y.npy" in file:
            y = np.load(file)
            break

    if y is None:
        raise ValueError("No 'y.npy' file found in the specified files.")

    for file in files:
        if "y.npy" not in file:
            match_ = re.search(r"y_hat_(.*?)\.npy", file)
            if match_:
                model = match_.group(1)
                print(f"Processing model: {model}")

                y_hat = np.load(file)
                try:
                    mae, mape, rmse = report_household_level(y, y_hat)
                    mae_agg, mape_agg, rmse_agg = report_aggregate_level(y, y_hat)
                except Exception as e:
                    print(f'Error at model: {model}, Error: {e}')
                    continue

                # Add metrics to DataFrame
                df.loc['MAE', model] = mae
                df.loc['MAPE', model] = mape
                df.loc['RMSE', model] = rmse

                df_agg.loc['MAE', model] = mae_agg
                df_agg.loc['MAPE', model] = mape_agg
                df_agg.loc['RMSE', model] = rmse_agg

    # Save the updated DataFrame back to the CSV file
    df.to_csv(os.path.join(folder_path, 'metrics.csv'))
    df_agg.to_csv(os.path.join(folder_path, 'agg_metrics.csv'))
    print(f"Metrics saved in {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process forecast value folders and nested folders for metrics.')
    parser.add_argument('folder', help='Root folder containing records for test periods and nested folders.')
    args = parser.parse_args()

    root_folder = args.folder

    # Initialize metrics DataFrames
    df = pd.DataFrame()
    df_agg = pd.DataFrame()

    # Process root folder
    print(f"Processing root folder: {root_folder}")
    process_folder(root_folder, df, df_agg)

    # Recursively process all subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            nested_folder = os.path.join(dirpath, dirname)
            print(f"Processing nested folder: {nested_folder}")

            # Reload or initialize metrics DataFrames for the nested folder
            df_nested = pd.DataFrame()
            df_agg_nested = pd.DataFrame()

            process_folder(nested_folder, df_nested, df_agg_nested)

    print(f"All folders and subfolders processed successfully.")
