from glob import glob
import os
import pandas as pd
import numpy as np
import argparse
import re
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
    # os.chdir(folder_path)
   
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
                if model not in df.columns:
                    df[model] = None
                df.loc['MAE', model] = mae
                df.loc['MAPE', model] = mape
                df.loc['RMSE', model] = rmse

                if model not in df_agg.columns:
                    df_agg[model] = None
                df_agg.loc['MAE', model] = mae_agg
                df_agg.loc['MAPE', model] = mape_agg
                df_agg.loc['RMSE', model] = rmse_agg

    # Save the updated DataFrame back to the CSV file
    df.to_csv(os.path.join(folder_path, 'metrics.csv'), index=False)
    df_agg.to_csv(os.path.join(folder_path, 'agg_metrics.csv'), index=False)
    print(f"Metrics saved in {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process forecast value folders and nested folders for metrics.')
    parser.add_argument('folder', help='Root folder containing records for test periods and nested folders.')
    args = parser.parse_args()

    root_folder = args.folder
    
    df = pd.read_csv(os.path.join(root_folder, 'metrics.csv'), index_col=0) if Path(os.path.join(root_folder, 'metrics.csv')).exists() else pd.DataFrame()
    df_agg = pd.read_csv(os.path.join(root_folder, 'agg_metrics.csv'), index_col=0) if Path(os.path.join(root_folder, 'agg_metrics.csv')).exists() else pd.DataFrame()

    # Process root folder
    print(f"Processing root folder: {root_folder}")
    try:
        process_folder(root_folder, df, df_agg)
    except:
        print("Hello!")
    # Recursively process all subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        print(dirpath)
        for dirname in dirnames:
            print(dirname)
            nested_folder = os.path.join(dirpath, dirname)
            print(f"Processing nested folder: {nested_folder}")

            # Reload metrics DataFrames for nested folder
            df_nested = pd.read_csv(os.path.join(nested_folder, 'metrics.csv'), index_col=0) if Path(os.path.join(nested_folder, 'metrics.csv')).exists() else pd.DataFrame()
            df_agg_nested = pd.read_csv(os.path.join(nested_folder, 'agg_metrics.csv'), index_col=0) if Path(os.path.join(nested_folder, 'agg_metrics.csv')).exists() else pd.DataFrame()
            
            process_folder(nested_folder, df_nested, df_agg_nested)

    print(f"All folders and subfolders processed successfully.")
