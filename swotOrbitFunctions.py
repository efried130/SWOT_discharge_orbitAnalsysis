# Master Functions for Plotting and Metrics
# Built for dictionary of dfs


import datetime
import time
import pathlib
import os,sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
import numpy as np
import pandas as pd
import cartopy
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import folium
import requests
from io import StringIO
import h5py
import math
import zipfile
from datetime import timedelta
import json
import seaborn as sns 
import warnings
from tqdm import tqdm 
from scipy.stats import pearsonr

def calc_cons(df):
    df = df[df['time'] != 'no_data']
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    if 'consensus' not in df['algo'].unique():
            algo_Q_cons_values = (
                df[df['algo'] != 'gauge']
                .groupby(['reach_id', 'time'])['Q']
                .median()
                .reset_index()
            )
            algo_Q_cons_values['algo'] = 'consensus'
            df = pd.concat([df, algo_Q_cons_values], ignore_index=True)
    return df

##############
## GET METRICS
##############


def calculate_metrics(df, reaches):
    """
    Calculate performance metrics for all algorithms, including 'algo_Q_cons',
    grouped by 'reach_id' and 'algo' from combined DataFrames.

    Parameters:
    df (pd.DataFrame): DataFrame containing discharge data
    reaches (list): List of unique reach IDs to evaluate

    Returns:
    pd.DataFrame: A DataFrame with calculated metrics for each 'reach_id' and 'algo'
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['Q'] > 1) & (df['Q'] < 1e7)]

    # Add 'consensus' if missing
    if 'consensus' not in df['algo'].unique():
        algo_Q_cons_values = (
            df[df['algo'] != 'gauge']
            .groupby(['reach_id', 'time'])['Q']
            .median()
            .reset_index()
        )
        algo_Q_cons_values['algo'] = 'consensus'
        df = pd.concat([df, algo_Q_cons_values], ignore_index=True)

    metrics_list = []

    for reach_id in reaches:
        reach_df = df[df['reach_id'] == reach_id]
        gauge_df = reach_df[reach_df['algo'] == 'gauge'][['time', 'Q']].set_index('time')

        for algo in reach_df['algo'].unique():
            if algo == 'gauge':
                continue

            algo_df = reach_df[reach_df['algo'] == algo][['time', 'Q']].set_index('time')
            aligned_df = gauge_df.join(algo_df, lsuffix='_gauge', rsuffix=f'_{algo}', how='inner').dropna()

            if aligned_df.empty or len(aligned_df) < 10:
                continue

            GQ = aligned_df['Q_gauge']
            SQ = aligned_df[f'Q_{algo}']

            NSE = 1 - (np.sum((GQ - SQ) ** 2) / np.sum((GQ - np.mean(GQ)) ** 2)) if np.sum((GQ - np.mean(GQ)) ** 2) != 0 else np.nan
            r, _ = pearsonr(SQ, GQ)  # Changed to Pearson's r
            KGE = 1 - np.sqrt(
                (r - 1) ** 2 +
                ((np.std(SQ) / np.std(GQ)) - 1) ** 2 +
                ((np.mean(SQ) / np.mean(GQ)) - 1) ** 2
            ) if not np.isnan(r) else np.nan
            RMSE = np.sqrt(np.mean((SQ - GQ) ** 2)) if len(GQ) > 0 else np.nan
            nRMSE = RMSE / np.mean(GQ) if np.mean(GQ) != 0 else np.nan
            nBIAS = (np.sum(SQ - GQ) / len(GQ)) / np.mean(GQ) if np.mean(GQ) != 0 else np.nan
            rRMSE = np.sqrt(nRMSE ** 2 - nBIAS ** 2) if not np.isnan(nRMSE) and not np.isnan(nBIAS) else np.nan
            norm_res = np.abs((SQ - GQ) / np.mean(GQ)) if np.mean(GQ) != 0 else np.nan
            sigma1 = np.nanpercentile(norm_res, 67) if isinstance(norm_res, pd.Series) else np.nan
            res = p.abs((SQ - GQ)) if np.mean(GQ) != 0 else np.nan
            res_67 = np.nanpercentile(res, 67) if isinstance(res, pd.Series) else np.nan
            one_sigma = res_67 / np.mean(GQ) if np.mean(GQ) != 0 else np.nan
            
            metrics_list.append({
                'reach_id': reach_id,
                'algo': algo,
                'NSE': NSE,
                'r': r,
                'KGE': KGE,
                'RMSE': RMSE,
                'nRMSE': nRMSE,
                'nBIAS': nBIAS,
                'rRMSE': rRMSE,
                '1-sigma': one_sigma,
                'n': len(GQ)
            })

    metrics_df = pd.DataFrame(metrics_list)
    df = df.merge(metrics_df, on=['reach_id', 'algo'], how='left')
    return df


# ADD COEFF OF VARIATION

def coeffVar(df, reach_list):
    # Filter the DataFrame by the list of reaches
    filtered_df = df[df['reach_id'].isin(reach_list)]

    # Group by 'reach_id' and 'algorithm' and calculate mean, sd, and sd/mean for 'algo_Q'
    grouped = filtered_df.groupby(['reach_id', 'algo']).agg(
        mean_algo_Q=('Q', 'mean'),
        sd_algo_Q=('Q', 'std')
    ).reset_index()

    # Calculate sd/mean and add it to the DataFrame
    # Avoid division by zero — assign NaN where mean is zero
    grouped['CV'] = np.where(
        grouped['mean_algo_Q'] == 0,
        np.nan,
        grouped['sd_algo_Q'] / grouped['mean_algo_Q']
)
    
    
    # Group by 'reach_id' and 'algorithm' and calculate mean, sd, and sd/mean for 'algo_Q'
    grouped_cons = filtered_df[filtered_df['algo'] == 'consensus'].drop_duplicates(subset=['reach_id', 'time', 'Q']).groupby(['reach_id']).agg(
        mean_cons_Q=('Q', 'mean'),
        sd_cons_Q=('Q', 'std')
    ).reset_index()    
    
    grouped_cons['CV_cons'] = grouped_cons['sd_cons_Q'] / grouped_cons['mean_cons_Q']

    # Group by 'reach_id' and 'algorithm' and calculate mean, sd, and sd/mean for 'algo_Q'
    grouped_gauge = filtered_df[filtered_df['algo'] == 'gauge'].drop_duplicates(subset=['reach_id', 'time', 'Q']).groupby(['reach_id']).agg(
        mean_gauge_Q=('Q', 'mean'),
        sd_gauge_Q=('Q', 'std')
    ).reset_index().dropna()
    
    grouped_gauge['CV_gauge'] = grouped_gauge['sd_gauge_Q'] / grouped_gauge['mean_gauge_Q']
    
    # Merge df1 and df2 on 'reach_id'
    merged_df = pd.merge(grouped, grouped_cons, on='reach_id', how='left')

    # Merge the resulting DataFrame with df3 on 'reach_id'
    final_df = pd.merge(merged_df, grouped_gauge, on='reach_id', how='left')
    
    return round(final_df, 3) #grouped, grouped_cons, grouped_gauge

# ADD RELATIVE MEAND DIFFERENCE

def relative_mean_difference(series1, series2):
    """
    Calculate the relative mean difference between two time series.

    Parameters:
    series1 (list or np.array): The first time series data.
    series2 (list or np.array): The second time series data (reference).

    Returns:
    float: The relative mean difference between the two time series.
    """
    # Convert to numpy arrays for vectorized operations
    series1 = np.array(series1, dtype=np.float64)
    series2 = np.array(series2, dtype=np.float64)

    # Check for empty or NaN-only series
    if series1.size == 0 or series2.size == 0 or np.all(np.isnan(series1)) or np.all(np.isnan(series2)):
        return np.nan  # Return NaN instead of triggering a warning

    # Calculate relative differences
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # Suppress warnings
        relative_differences = np.abs(series1 - series2) / series2

        # Handle divisions by zero
        relative_differences = np.where(np.isfinite(relative_differences), relative_differences, np.nan)

        # Compute mean safely
        rmd = np.nanmean(relative_differences)  # Ignore NaNs

    return rmd


def process_and_plot_rmd(df):
    """
    Process the DataFrame to calculate RMD for each reach_id and plot the CDF.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: A DataFrame with reach_id, algo, and corresponding RMD values.
    """
    # Ensure required columns are present
    required_columns = ['reach_id', 'algo', 'Q']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is not present in the DataFrame.")
    
    results = []
    reach_ids = df['reach_id'].unique()

    for reach_id in tqdm(reach_ids, desc="Processing Reaches RMD", unit="reach"):
        reach_df = df[df['reach_id'] == reach_id]
        
        # Filter out if not enough unique algorithms or discharge values
        filtered_reach_df = reach_df[reach_df['algo'] != 'consensus']
        if filtered_reach_df['algo'].nunique() < 3 or reach_df['Q'].nunique() < 3:
            #print('Not enough algorithms to calculate RMD')
            continue

        for algo in reach_df['algo'].unique():
            algo_df = reach_df[reach_df['algo'] == algo]
            algo_cons = reach_df[reach_df['algo'] == 'consensus']
            algo_gauge = reach_df[reach_df['algo'] == 'gauge']
            
            # Merge dataframes on 'time'
            merged_df = pd.merge(algo_df[['time', 'Q']], algo_cons[['time', 'Q']], on='time', suffixes=('_algo', '_cons'))
            #merged_df_gauge = pd.merge(algo_df[['time', 'Q']], algo_gauge[['time', 'Q']], on='time', suffixes=('_algoG', '_gauge'))

            if merged_df.empty: #or merged_df_gauge.empty:
                #print('No overlapping data between algo and consensus')
                continue  # Skip if there's no overlapping data

            # Calculate RMD safely
            rmd = relative_mean_difference(merged_df['Q_algo'], merged_df['Q_cons'])
            #rmd_G = relative_mean_difference(merged_df_gauge['Q_algoG'], merged_df_gauge['Q_gauge'])

            results.append({'reach_id': reach_id, 'algo': algo, 'RMD_cons': rmd}) #, 'RMD_gauge': rmd_G})

    results_df = pd.DataFrame(results)

    # Replace infinities and drop NaNs
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    results_df.dropna(subset=['RMD_cons'], inplace=True) #, 'RMD_gauge'

    return results_df


def append_RMD(dfs_q):
    """
    Append the Relative Mean Difference (RMD) to each DataFrame in dfs_q.

    Parameters:
    dfs_q (dict): Dictionary containing DataFrames to be updated.

    Returns:
    dict: Updated dfs_q with appended RMD values.
    """
    for label, df in dfs_q.items():
        if df.empty:
            print('EMPTY DF')
            continue

        # Process and get RMD DataFrame
        results_df = process_and_plot_rmd(df=df)

        # Merge RMD values back into the original DataFrame
        if not results_df.empty:
            df = df.merge(
                results_df[['reach_id', 'algo', 'RMD_cons']], #, 'RMD_gauge']],
                on=['reach_id', 'algo'],
                how='left'
            )

        # Update the DataFrame in the dictionary
        dfs_q[label] = df

    return dfs_q

#PLOT RMD

def cdfPlot_RMD(dfs_q, color_dict, column_to_plot):
    """
    Plot the CDF of RMD for each DataFrame in the dfs_q dictionary.

    Parameters:
    dfs_q (dict): Dictionary containing DataFrames to plot.
    """
    for label, df in dfs_q.items():
        if df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))


        # Plot CDF for each algorithm if it exists in the DataFrame
        for algo, color in color_dict.items():
            algo_data = df[df['algo'] == algo]
            if algo_data.empty:
                continue


            # Sort the data and compute CDF values
            x = np.sort(algo_data[column_to_plot].dropna())
            y = np.arange(1, len(x) + 1) / float(len(x))

            # Plot CDF
            ax = sns.lineplot(x=x, y=y, linewidth=3, color=color, 
                               label=f'{algo} (n={algo_data.reach_id.nunique()})', errorbar=None)

        # Plot formatting
        ax.set_xlabel(f'{label} {column_to_plot}', fontsize=30)
        ax.set_ylabel('Proportion', fontsize=30)
        ax.axhline(y=0.67, color='grey', linestyle='--')
        plt.xticks(fontsize=26, rotation=45)
        plt.yticks(fontsize=26)
        plt.xlim([0, 3])
        plt.gca().tick_params(axis='y', pad=15)
        plt.legend(loc='lower right', ncol=1, fontsize=26)
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()
        
        
# PLOT CDF of CV

def append_coeffVar(dfs_q):
    for label, df in dfs_q.items():
        if df.empty:
            continue

        # Group by 'reach_id' and 'algo' and calculate mean, sd, and CV for 'Q'
        coeff_df = coeffVar(df, df[df['algo']=='consensus'].reach_id.unique())

        # Merge coefficient of variation values back into the original DataFrame
        df = df.merge(
            coeff_df[['reach_id', 'algo', 'CV', 'CV_cons', 'CV_gauge']],
            on=['reach_id', 'algo'], how='left'
        )

        # Update the DataFrame in the dictionary
        dfs_q[label] = df

    return dfs_q
#Gauge CV
def plot_cdf_coeff(dfs_q, color_dict, algos_to_plot):
    """
    Plot the CDF of Coefficient of Variation (CV and CV_gauge) for each DataFrame in the dictionary.

    Parameters:
    dfs_q (dict): Dictionary containing DataFrames to plot.
    color_dict (dict): Mapping of algorithm names to their respective colors.
    """

    for label, df in dfs_q.items():
        if df.empty:
            continue

        plt.figure(figsize=(15, 8))

        # Plot known algos
        for algo in algos_to_plot:
            algo_df = df[df['algo'] == algo]
            if algo_df.empty or 'CV' not in algo_df:
                continue

            coeff_var_algo_sorted = np.sort(algo_df['CV'].dropna())
            cdf = np.arange(1, len(coeff_var_algo_sorted) + 1) / len(coeff_var_algo_sorted)
            
            median_val = np.round(np.median(algo_df['CV']), 2)
            percentile_68 = np.round(np.percentile(algo_df['CV'], 68), 2)
            
            
            plt.plot(
                coeff_var_algo_sorted,
                cdf,
                label=f'{algo} (n={len(algo_df.reach_id.unique())}, 68_perc={percentile_68})',
                linewidth=6 if algo in ['consensus', 'gauge'] else 3,
                color=color_dict.get(algo, 'black'),
                linestyle='-.' if algo == 'gauge_swot_match' else '-'
            )


        # Plot gauge CDF
#         if 'CV_gauge' in df:
#             coeff_var_gauge_sorted = np.sort(df['CV_gauge'].dropna())
#             p_gauge = np.arange(1, len(coeff_var_gauge_sorted) + 1) / len(coeff_var_gauge_sorted)
            
#             median_val = np.round(np.median(algo_df['CV_gauge']), 2)
#             percentile_68 = np.round(np.percentile(algo_df['CV_gauge'], 68), 2)
            
#             plt.plot(
#                 coeff_var_gauge_sorted,
#                 p_gauge,
#                 label=f'Gauge_all (n={len(df[df["algo"] == "gauge"].reach_id.unique())}, 68_perc={percentile_68})',
#                 linewidth=6,
#                 color='black',
#                 linestyle='-.'
#             )

        # Plot customization
        plt.hlines(y=0.66, xmin=0, xmax=10, color='black', linestyle='--', linewidth=3)
        plt.xlabel(f'Coefficient of Variation ({label})', fontsize=30)
        plt.ylabel('Proportion', fontsize=30)
        plt.xticks(np.arange(0, 3.6, 0.25), fontsize=16, rotation=45)
        plt.yticks(fontsize=26, rotation=45)
        plt.gca().tick_params(axis='y', pad=15)
        plt.legend(loc='lower right', fontsize=26)
        plt.grid(True)
        plt.tight_layout()
        plt.xlim([0, 2.5])

        plt.show()
        
        
        
        
        
#OTHER DISPERSION METRICS BASED ON IQR
def compute_iqr(series):
    """Raw Interquartile Range = Q3 - Q1"""
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    return q3 - q1


def compute_normalized_IQR(series):
    """Normalized IQR = (Q3 - Q1) / median"""
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    median = np.nanmedian(series)
    if median != 0 and not np.isnan(median):
        return (q3 - q1) / median
    return np.nan

def compute_normalized_range(series):
    """Normalized range = (max - min) / median"""
    min_val = np.nanmin(series)
    max_val = np.nanmax(series)
    median = np.nanmedian(series)
    if median != 0 and not np.isnan(median):
        return (max_val - min_val) / median
    return np.nan

def compute_cqv(series):
    """Coefficient of Quartile Variation = (Q3 - Q1) / (Q3 + Q1)"""
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    denom = q3 + q1
    if denom != 0 and not np.isnan(denom):
        return (q3 - q1) / denom
    return np.nan

def add_all_variability_metrics_to_dict(df_dict):
    """
    Takes a dictionary of DataFrames and appends variability metrics
    to each, grouped by 'reach_id' and 'algo'.
    Returns a new dictionary with updated DataFrames.
    """
    result = {}

    for name, df in df_dict.items():
        grouped = (
            df.groupby(['reach_id', 'algo'])
            .agg(
                norm_IQR=('Q', compute_normalized_IQR),
                norm_range=('Q', compute_normalized_range),
                CQV=('Q', compute_cqv),
                IQR=('Q', compute_iqr)
            )
            .reset_index()
        )

        df_merged = df.merge(grouped, on=['reach_id', 'algo'], how='left')
        result[name] = df_merged

    return result


def plot_all_variability_cdfs(dfs_q, color_dict, metrics):
    """
    Plot the CDFs for all specified variability metrics from a dictionary of DataFrames.

    Parameters:
    - dfs_q (dict): Dictionary where keys are labels and values are DataFrames.
    - color_dict (dict): Mapping of algorithm names to colors.
    - metrics (list): List of variability metrics to plot. Defaults to all supported metrics.
    """
    for label, df in dfs_q.items():
        if df.empty:
            continue

        for metric in metrics:
            if metric not in df.columns:
                continue

            plt.figure(figsize=(12, 8))

            for algo in ['hivdi', 'sic4dvar', 'momma', 'neobam', 'consensus', 'geobam', 'metroman', 'gauge', 'gauge_swot_match']:
                algo_df = df[df['algo'] == algo]
                if algo_df.empty or metric not in algo_df:
                    continue

                # Filter to only reach_ids with at least 10 rows
                valid_reaches = algo_df.groupby('reach_id').filter(lambda x: len(x) >= 10)
                if valid_reaches.empty:
                    continue

                values = valid_reaches[metric].dropna()
                sorted_vals = np.sort(values)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

                
                plt.plot(
                    sorted_vals, cdf, 
                    label=f'{algo} (n={len(algo_df.reach_id.unique())})',
                    linewidth=6 if algo in ['consensus', 'gauge'] else 3, linestyle='-.' if algo == 'gauge_swot_match' else '-', color=color_dict.get(algo, 'black')
                )

            # Plot customization
            plt.hlines(y=0.66, xmin=0, xmax=10, color='black', linestyle='--', linewidth=3)
            plt.xlabel(f'{metric} ({label})', fontsize=28)
            plt.ylabel('Proportion', fontsize=28)
            plt.xticks(fontsize=18, rotation=45)
            plt.yticks(fontsize=24)
            plt.tick_params(axis='y', pad=15)
            plt.legend(loc='lower right', fontsize=20)
            plt.grid(True)
            plt.tight_layout()

            # Metric-specific x-limits (optional)
            if metric == 'norm_IQR':
                plt.xlim([0, 5])
                plt.xticks(np.arange(0, 5.1, 0.5), fontsize=14, rotation=45)
            elif metric == 'norm_range':
                plt.xlim([0,10])
            elif metric == 'CQV':
                plt.xlim([0,1])
            elif metric == 'IQR':
                plt.xlim([0,2000])

            plt.title(f'CDF of {metric} for {label}', fontsize=30)
            plt.show()
