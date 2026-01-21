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

            
def remove_low_cv_and_recalc_consensus(dfs_dict, CV_thresh):
    """
    For each DataFrame in the input dictionary:
    - Removes rows where CV < threshold
    - Drops existing 'consensus' rows
    - Recalculates consensus (excluding 'geobam') and appends it

    Parameters:
    - dfs_dict (dict): Dictionary of DataFrames keyed by label.
    - cv_col (str): Name of the CV column (default ' CV').

    Returns:
    - dict: Dictionary of processed DataFrames.
    """

    cleaned_dict = {}
    for label, df in dfs_dict.items():
        df = df.copy()
        #df = df[df['algo'] != 'geobam']
        df = df[df['CV'] > CV_thresh]
        df = df[df['algo'] != 'consensus']
        df = calc_cons(df)
        cleaned_dict[label] = df

    return cleaned_dict


def get_season_orbits(date):
    month = date.month
    day = date.day
    if (month == 3 and day >= 31) or (month in [4, 5, 6]) or (month == 7 and day < 12):
        return '4-7'
    elif (month == 7 and day >= 12) or (month in [7, 8]) or (month == 9 and day < 30):
        return '7-10'
    elif (month == 9 and day >= 30) or (month in [10, 11]):
        return '10-1'
    else:
        return '1-4'

def tukey_filter(df, column, threshold=1.5):
    """
    Apply Tukey outlier filtering to a specific column in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - column: column name (string) to apply Tukey filter on
    - threshold: float, the multiplier for the IQR (default is 1.5)
    
    Returns:
    - Filtered DataFrame where values are within [Q1 - threshold*IQR, Q3 + threshold*IQR]
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calc_cons(df):
    if 'time_str' in df.columns:
        df = df[df['time_str'] != 'no_data']
    if 'time' not in df.columns:
        print('NO TIME COLUMN FOUND')
        
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

def modified_z_filter(series, threshold=3.5):
    median = np.median(series)
    mad = median_abs_deviation(series, scale='normal')
    if mad == 0:
        return series
    mzs = 0.6745 * (series - median) / mad
    return series[np.abs(mzs) <= threshold]


# CV SUMMARY FIGURES


def plot_metric_cdfs_with_filters(df, algo, algo_threshold, algo_threshold2):
    metric_cols = ['r', 'nBIAS', 'NSE', '1-sigma']
    colors = {'r': 'tab:blue', '1-sigma': 'tab:orange', 'nBIAS': 'tab:green', 'NSE': 'tab:red'}
    
    # Metric display names
    metric_display = {'r': 'r', '1-sigma': '1-sigma', 'nBIAS': '|nBIAS|', 'NSE': 'NSE'}

    # --- Read and compute baseline metrics once ---
    data_baseline = df
    algos_to_plot = ['hivdi', 'sic4dvar', 'momma', 'sad', 'neobam', 'consensus', 'geobam', 'metroman', 'gauge_swot_match','gauge']
    plot_cdf_coeff(dfs_q={'CV': data_baseline}, color_dict=color_dict, algos_to_plot = algos_to_plot)

    data_baseline = data_baseline[data_baseline['algo'] == algo]
    
    
    # --- Initialize rejected_data dict ---
    rejected_data = {}
    full_reach_ids = set(df['reach_id'].dropna().unique())

    # --- Filter once per dataset ---
    df_algo_cv = df[df['CV'] > algo_threshold]
    df_algo_cv_rejected = df[df['CV'] <= algo_threshold]
    df_algo_cv = df_algo_cv[df_algo_cv['algo'] != algo]
    df_algo_cv_cons_recalc = calc_cons(df_algo_cv)
    df_algo_cv_cons_recalc = df_algo_cv_cons_recalc.drop(columns=['CV', 'CV_cons', 'CV_gauge','RMD_cons'])
    df_algo_cv_cons_recalc_dict = append_coeffVar({'CV': df_algo_cv_cons_recalc})
    df_algo_cv_cons_recalc = df_algo_cv_cons_recalc_dict['CV']
    df_algo_cv_cons_metrics = calculate_metrics(
        df=df_algo_cv_cons_recalc[['algo', 'Q', 'time', 'gauge_discharge', 'gauge_time', 'reach_id']],
        reaches=list(df_algo_cv_cons_recalc["reach_id"].unique())
    )
    algos_to_plot = ['hivdi', 'sic4dvar', 'momma', 'sad', 'neobam', 'consensus', 'geobam', 'metroman', 'gauge_swot_match','gauge']
    plot_cdf_coeff(dfs_q={'CV': df_algo_cv_cons_recalc}, color_dict=color_dict, algos_to_plot = algos_to_plot)

    df_algo_cv_cons_recalc = df_algo_cv_cons_metrics[df_algo_cv_cons_metrics['algo'] == algo]
    
    
    # Store rejected data for first threshold
    filtered_reach_ids_cv = set(df_algo_cv_cons_recalc['reach_id'].dropna().unique())
    label_cv = f'CV > {algo_threshold}'
    rejected_ids_cv = full_reach_ids - filtered_reach_ids_cv
    if label_cv not in rejected_data:
        rejected_data[label_cv] = df_algo_cv_rejected
    
    # --- Read CV dataset ---
    df_algo_cv2 = df[df['CV'] > algo_threshold2]
    df_algo_cv2_rejected = df[df['CV'] <= algo_threshold2]
    df_algo_cv2 = df_algo_cv2[df_algo_cv2['algo'] != algo]
    df_algo_cv_cons_recalc2 = calc_cons(df_algo_cv2)
    df_algo_cv_cons_recalc2 = df_algo_cv_cons_recalc2.drop(columns=['CV', 'CV_cons', 'CV_gauge','RMD_cons'])
    df_algo_cv_cons_recalc2_dict = append_coeffVar({'CV': df_algo_cv_cons_recalc2})
    df_algo_cv_cons_recalc2 = df_algo_cv_cons_recalc2_dict['CV']
    df_algo_cv_cons_metrics2 = calculate_metrics(
        df=df_algo_cv_cons_recalc2[['algo', 'Q', 'time', 'gauge_discharge', 'gauge_time', 'reach_id']],
        reaches=list(df_algo_cv_cons_recalc2["reach_id"].unique())
    )
    algos_to_plot = ['hivdi', 'sic4dvar', 'momma', 'sad', 'neobam', 'consensus', 'geobam', 'metroman', 'gauge_swot_match','gauge']
    plot_cdf_coeff(dfs_q={'CV': df_algo_cv_cons_recalc2}, color_dict=color_dict, algos_to_plot = algos_to_plot)

    df_algo_cv_cons_recalc2 = df_algo_cv_cons_metrics2[df_algo_cv_cons_metrics2['algo'] == algo]
        
    # Store rejected data for second threshold
    filtered_reach_ids_cv2 = set(df_algo_cv_cons_recalc2['reach_id'].dropna().unique())
    label_cv2 = f'CV > {algo_threshold2}'
    rejected_ids_cv2 = full_reach_ids - filtered_reach_ids_cv2
    if label_cv2 not in rejected_data:
        rejected_data[label_cv2] = df_algo_cv2_rejected
    
    # --- Helper for percentile summaries ---
    def metric_summary(label, df_metrics, metric):
        df_metrics_reach = df_metrics.drop_duplicates(subset='reach_id')  
        vals = df_metrics_reach[metric].abs().dropna() if metric == 'nBIAS' else df_metrics_reach[metric].dropna()
        if vals.empty:
            return None
        return {
            'filter_stage': label,
            'algo': algo,
            'metric': metric_display[metric],  # Use display name
            'p32': np.percentile(vals, 32).round(2),
            'median': np.median(vals).round(2),
            'p67': np.percentile(vals, 67).round(2),
            'n_reaches': df_metrics_reach.loc[vals.index, 'reach_id'].nunique(),
            'obs/reach': df_metrics.reach_id.value_counts().median()
        }

    # --- Collect summaries efficiently ---
    summary_rows = []
    summary_info = {}  # store n_reaches & obs/reach for plotting

    for metric in metric_cols:
        for label, df_metrics in [
            ('baseline', data_baseline),
            (f'CV > {algo_threshold}', df_algo_cv_cons_recalc),
            (f'CV > {algo_threshold2}', df_algo_cv_cons_recalc2)
        ]:
            summary = metric_summary(label, df_metrics, metric)
            if summary:
                summary_rows.append(summary)
                summary_info[(metric, label)] = summary['n_reaches'], summary['obs/reach']

    summary_df = pd.DataFrame(summary_rows)

    # --- Compute percent retained ---
    original_n = (
        summary_df[summary_df['filter_stage'] == 'baseline']
        .set_index('algo')['n_reaches']
        .to_dict()
    )
    summary_df['percent_reaches_retained'] = summary_df.apply(
        lambda row: round((row['n_reaches'] / original_n[row['algo']]) * 100, 1)
        if row['algo'] in original_n and original_n[row['algo']] > 0 else np.nan,
        axis=1
    )

    # --- Plot all metrics together ---
    plt.figure(figsize=(12, 8))
    for metric in metric_cols:
        for stage, df_metrics, style in [
            ('baseline', data_baseline, '-'),
            (f'CV > {algo_threshold}', df_algo_cv_cons_recalc, '--'),
            (f'CV > {algo_threshold2}', df_algo_cv_cons_recalc2, ':')
        ]:
            df_metrics_reach = df_metrics.drop_duplicates(subset='reach_id')
            vals = df_metrics_reach[metric].abs().dropna() if metric == 'nBIAS' else df_metrics_reach[metric].dropna()            
            
            if vals.empty:
                continue
            x = np.sort(vals)
            y = np.arange(1, len(x) + 1) / len(x)

            # Use n_reaches and obs/reach from summary_info
            n_reaches, obs_per_reach = summary_info.get((metric, stage), (np.nan, np.nan))

            plt.plot(
                x, y, color=colors[metric], linewidth=3, linestyle=style,
                label=f'{metric_display[metric]} ({stage}, n={n_reaches}, obs/reach={obs_per_reach})'
            )

    # --- Styling ---
    for yline in [0.32, 0.5, 0.67]:
        plt.axhline(y=yline, color='black', linewidth=1.5, linestyle='--')
    plt.xlim(-0.5, 1)
    plt.ylim(0, 1)
    plt.xlabel("Metric value", fontsize=30)
    plt.ylabel("Proportion", fontsize=30)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('//figs/CV_analysis_gauge_orbit.pdf', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

    # --- NEW: Plot consensus CV only ---
    plt.figure(figsize=(12, 8))
    # First plot gauge CV (from baseline data)
    df_gauge = df[df['algo'] == 'gauge'].copy()
    df_gauge_reach = df_gauge.drop_duplicates(subset='reach_id')
    gauge_cv_vals = df_gauge_reach['CV_gauge'].dropna()
    
    if not gauge_cv_vals.empty:
        x = np.sort(gauge_cv_vals)
        y = np.arange(1, len(x) + 1) / len(x)
        n_reaches = df_gauge_reach['reach_id'].nunique()
        obs_per_reach = df_gauge.reach_id.value_counts().median()
        
        plt.plot(
            x, y, color='black', linewidth=3, linestyle='-',
            label=f'gauge (n={n_reaches}, obs/reach={obs_per_reach})'
        )
    for stage, df_metrics, style, label_suffix in [
        ('baseline', data_baseline, '-', 'baseline'),
        (f'CV > {algo_threshold}', df_algo_cv_cons_recalc_dict['CV'], '--', f'CV > {algo_threshold}'),
        (f'CV > {algo_threshold2}', df_algo_cv_cons_recalc2_dict['CV'], ':', f'CV > {algo_threshold2}')
    ]:
        print(stage)
        # Get consensus CV values
        df_consensus = df_metrics[df_metrics['algo'] == 'consensus'].copy()
        df_consensus_reach = df_consensus.drop_duplicates(subset='reach_id')
        cv_vals = df_consensus_reach.CV.dropna()
        
        if cv_vals.empty:
            continue
            
        x = np.sort(cv_vals)
        y = np.arange(1, len(x) + 1) / len(x)
        
        n_reaches = df_consensus_reach['reach_id'].nunique()
        obs_per_reach = df_consensus.reach_id.value_counts().median()
        
        plt.plot(
            x, y, color='sienna', linewidth=3, linestyle=style,
            label=f'{label_suffix} (n={n_reaches}, obs/reach={obs_per_reach})'
        )
    
    # --- Styling ---
    for yline in [0.32, 0.5, 0.67]:
        plt.axhline(y=yline, color='black', linewidth=1.5, linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Consensus CV", fontsize=30)
    plt.ylabel("Proportion", fontsize=30)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('//figs/CV_consensus_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

    return summary_df[['filter_stage', 'algo', 'metric', 'p32', 'median', 'p67',
                       'n_reaches', 'percent_reaches_retained', 'obs/reach']], rejected_data



def plot_reach_id_counts_by_algo(df_orig, df, filter_condition, color_dict, exclude_algos):
    """
    Plots a bar chart of the number of unique reach_ids per algorithm with value labels.
    Parameters:
        df (pd.DataFrame): Input DataFrame with at least 'algo' and 'reach_id' columns.
        color_dict (dict): Optional dict mapping algo names to colors.
        exclude_algos (list): Optional list of algos to exclude from the plot.
        filter_condition (str): Label for the filter condition.
    """
    # Drop excluded algorithms if specified
    if exclude_algos:
        df = df[~df['algo'].isin(exclude_algos)]
    
    # Count unique reach_ids per algo
    algo_counts = df.groupby('algo')['reach_id'].nunique().sort_values(ascending=False)
    
    # Calculate total reaches per algorithm (for percentage calculation)
    total_per_algo = df_orig.groupby('algo')['reach_id'].nunique()
    
    # Assign colors based on provided color_dict (default to gray)
    if color_dict is None:
        color_dict = {}
    bar_colors = [color_dict.get(algo, 'gray') for algo in algo_counts.index]
    
    # Plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(algo_counts.index, algo_counts.values, color=bar_colors)
    
    # Add value labels above each bar with percentage
    for bar, algo in zip(bars, algo_counts.index):
        height = bar.get_height()
        freq = int(height)
        total = total_per_algo[algo]
        pct = (freq / total * 100) if total > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, 
                 f'{freq} ({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=19)
    
    plt.xlabel(f'Algorithms', fontsize=26)
    plt.xticks(range(len(algo_counts)), [algo.capitalize() for algo in algo_counts.index], rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel(f'Frequency', fontsize=26)
    plt.title(f'Rejected Reach IDs Under {filter_condition}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.ylim([0, algo_counts.max() + 10])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f'//figs/CVrejectedAlgos_{filter_condition}.png', dpi=300)
    plt.show()

from datetime import timedelta

def plot_discharge_with_metrics(df_metrics, divide_date, color_dict):
    df_metrics = df_metrics.sort_values('reach_id')

    for reach_id in df_metrics['reach_id'].unique():
        df_reach = df_metrics[df_metrics['reach_id'] == reach_id].sort_values('time')

        if (df_reach['algo'] == 'gauge').sum() < 10:
            continue

        plt.figure(figsize=(10, 6))

        # Plot gauge discharge
        gauge_discharge = df_reach[df_reach['algo'] == 'gauge']['Q']
        gauge_time = pd.to_datetime(df_reach[df_reach['algo'] == 'gauge']['time'])
        if len(gauge_discharge) > 0:
            plt.scatter(gauge_time, gauge_discharge, label="Gauge", alpha=0.8, color='black')
            plt.plot(gauge_time, gauge_discharge, color='black', alpha=0.8, linestyle='--')

        # Plot each algorithm
        for algorithm in df_reach['algo'].unique():
            if algorithm == 'gauge':
                continue
            df_algo = df_reach[df_reach['algo'] == algorithm]
            discharge_time = pd.to_datetime(df_algo['time'])
            discharge_algo = df_algo['Q']
            color = color_dict.get(algorithm, 'black')

            if algorithm == 'consensus':
                plt.scatter(discharge_time, discharge_algo, alpha=1.0,
                            label=f"{algorithm.upper()}", color=color, marker='X')
                plt.plot(discharge_time, discharge_algo, color=color, alpha=1.0, linewidth=2.5)
            else:
                plt.scatter(discharge_time, discharge_algo, alpha=0.3,
                            label=f"{algorithm.upper()}", color=color)
                plt.plot(discharge_time, discharge_algo, color=color, alpha=0.3, linestyle='--')

        # Metrics annotation (based on consensus row if available)
        df_algo_q_cons = df_reach[df_reach['algo'] == 'consensus']
        if not df_algo_q_cons.empty:
            metrics = df_algo_q_cons.iloc[0]

            text_x = pd.to_datetime(df_algo_q_cons['time']).max() + timedelta(50)
            text_y = df_reach['Q'].min() / 3.5

            metrics_text = (
                f"R: {metrics['r']:.2f}\n"
                f"NSE: {metrics['NSE']:.2f}\n"
                #f"KGE: {metrics['KGE']:.2f}\n"
                #f"RMSE: {metrics['RMSE']:.2f}\n"
                #f"nRMSE: {metrics['nRMSE']:.2f}\n"
                f"|nBIAS|: {abs(metrics['nBIAS']):.2f}\n"
                #f"rRMSE: {metrics['rRMSE']:.2f}\n"
                f"1-sigma: {metrics['1-sigma']:.2f}\n"
                f"n: {metrics['n']}\n"
                #f"CV_gauge: {metrics['CV_gauge']:.2f}"
            )

            # --- Add CV per algorithm if available ---
            for algo in df_reach['algo'].unique():
                df_algo = df_reach[df_reach['algo'] == algo]
                if 'CV' in df_algo.columns and not df_algo['CV'].isna().all():
                    cv_val = df_algo['CV'].dropna().iloc[0]
                    metrics_text += f"CV {algo}: {cv_val:.2f}\n"

            plt.text(text_x, text_y, metrics_text,
                     fontsize=12, ha='left', va='bottom', color='black')

        # Final plot setup
        plt.vlines(x=divide_date, ymin=0, ymax=df_reach.Q.max(),
                   colors='black', linestyle='-')
        plt.suptitle(f"Discharge for Reach ID: {reach_id}", fontsize=18)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Discharge (m³/s)', fontsize=20)
        plt.xticks(rotation=45, fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

def plot_quad_consensus(dfs_dict, labels, divide_date, color_dict, algo, reach_ids):
    """
    Plot a 2x2 grid of hydrographs for four specified reach_ids with subplot labels a-d.

    Parameters
    ----------
    dfs_dict : dict
        Dictionary of DataFrames keyed by labels.
    labels : list
        List of keys from dfs_dict to plot.
    divide_date : pd.Timestamp
        Date to separate FS vs Science orbit.
    color_dict : dict
        Color mapping for each label.
    algo : str
        Algorithm to plot.
    reach_ids : list
        List of exactly four reach_ids to plot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    assert len(reach_ids) == 4, "Must provide exactly four reach_ids"

    marker_styles = ['o', 'd', '^', 'v', '*', 'P', 'h', '8']
    subplot_labels = ['a', 'b', 'c', 'd']

    fig, axes = plt.subplots(2, 2, figsize=(25, 18), sharex=False, sharey=False)
    axes = axes.flatten()

    plot_order = ['gauge', 'Continuous', 'Science', 'Fast', 'Sampled']
    point_sizes = {
        'gauge': 250,
        'Continuous': 240,
        'Science': 150,
        'Fast': 150,
        'Sampled': 240
    }

    for i, (ax, reach_id) in enumerate(zip(axes, reach_ids)):
        max_Q_all = []

        df_gauge = dfs_dict.get('Continuous')
        df_gauge_reach = df_gauge[(df_gauge['reach_id'] == reach_id) &
                                  (df_gauge['algo'] == 'gauge')].copy() if df_gauge is not None else None

        if df_gauge_reach is not None and not df_gauge_reach.empty:
            df_gauge_reach['time'] = pd.to_datetime(df_gauge_reach['time'])
            df_gauge_reach = df_gauge_reach.sort_values('time')
            all_dates = df_gauge_reach['time'].drop_duplicates().reset_index(drop=True)
            master_idx = range(len(all_dates))
            date_to_idx = dict(zip(all_dates, master_idx))
            df_gauge_reach['seq_idx'] = df_gauge_reach['time'].map(date_to_idx)
            df_gauge_reach = df_gauge_reach.dropna(subset=['seq_idx'])

            ax.plot(df_gauge_reach['seq_idx'].values, df_gauge_reach['Q'].values,
                    label="GAUGE", color=color_dict.get('gauge', 'black'),
                    alpha=0.5, linewidth=4, zorder=1)
            max_Q_all.append(df_gauge_reach['Q'].max())

        for label in plot_order:
            if label == 'gauge' or label not in dfs_dict:
                continue

            df = dfs_dict[label]
            color = color_dict.get(label, 'C0')
            marker = marker_styles[labels.index(label) % len(marker_styles)]
            df_reach = df[(df['reach_id'] == reach_id) & (df['algo'] == algo)].copy()
            if df_reach.empty:
                continue

            df_reach['time'] = pd.to_datetime(df_reach['time'])
            df_reach['seq_idx'] = df_reach['time'].map(date_to_idx)
            df_reach = df_reach.dropna(subset=['seq_idx'])

            if label == 'Continuous' and df_reach['time'].min() > divide_date:
                continue

            Q = df_reach['Q'].values
            x_vals = df_reach['seq_idx'].values.astype(int)
            max_Q_all.append(Q.max())

            ax.scatter(x_vals, Q, label=f"{label.upper()} {algo}",
                       alpha=1.0, color=color, marker=marker, s=point_sizes[label])

        divide_idx = (all_dates < divide_date).sum() if df_gauge_reach is not None else 0
        if divide_idx > 0 and len(max_Q_all) > 0:
            ymax = max(max_Q_all) + 0.1*max(max_Q_all)
            ax.vlines(x=divide_idx, ymin=0, ymax=ymax, colors='black', linewidth=5, linestyle='--')
            ax.text(divide_idx-(divide_idx/2.3), ymax - 0.1*max(max_Q_all), "FSO",
                    ha='center', va='bottom', fontsize=24, fontweight='bold')
            ax.text(divide_idx + (divide_idx/1.6), ymax - 0.1*max(max_Q_all), "SO",
                    ha='center', va='bottom', fontsize=24, fontweight='bold')

        # Add subplot letter label
        # ax.text(0.02, 0.92, subplot_labels[i], transform=ax.transAxes,
        #         fontsize=36, fontweight='bold', va='top', ha='left')

        ax.set_title(f"Reach ID: {reach_id}", fontsize=32)
        ax.set_xlabel('Sequential Observations', fontsize=30)
        ax.set_ylabel('Discharge (m³/s)', fontsize=30)
        ax.tick_params(axis='both', labelsize=28)

        if i == 0:
            ax.legend(fontsize=22)

    plt.tight_layout()
    plt.savefig(f'//figs/hydrograph_quad.pdf',
                bbox_inches='tight', dpi=350)
    plt.show()


    
def plot_consensus_from_multiple_dfs(dfs_dict, labels, divide_date, color_dict, algo):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Ensure labels is a list (in case dict_keys is passed)
    labels = list(labels)

    # Assert input integrity
    for label in labels:
        if label == 'gauge':
            continue
        assert label in dfs_dict, f"Label '{label}' not found in dfs_dict"

    # Distinct markers
    marker_styles = ['o', 'o', 'o', 'd', '^', 'v', '*', 'P', 'h', '8']

    # Step 1: Filter reach_ids with data before divide_date in Continuous
    reach_ids_before_divide = set()
    df_continuous = dfs_dict.get('Continuous')
    if df_continuous is not None:
        df_continuous_algo = df_continuous[df_continuous['algo'] == algo].copy()
        df_continuous_algo['time'] = pd.to_datetime(df_continuous_algo['time'])
        for rid, group in df_continuous_algo.groupby('reach_id'):
            if (group['time'] < divide_date).any():
                reach_ids_before_divide.add(rid)

    # Step 2: Get ALL reach_ids across all labels (union instead of intersection)
    all_reach_ids = set()
    for label in labels:
        if label == 'gauge':
            continue
        reach_ids = set(dfs_dict[label][dfs_dict[label]['algo'] == algo]['reach_id'].unique())
        all_reach_ids.update(reach_ids)
        print(f"{label}: {len(reach_ids)} reach_ids")
    
#     # Filter to only those with data before divide_date
#     all_reach_ids = sorted(all_reach_ids.intersection(reach_ids_before_divide))
    all_reach_ids = sorted(all_reach_ids)
    print(f"\nTotal reach_ids to plot: {len(all_reach_ids)}")
    print(f"Reach IDs: {all_reach_ids}")

    # Define plotting order and sizes
    plot_order = ['gauge', 'Continuous', 'Science', 'Fast', 'Sampled']
    point_sizes = {
        'gauge': 250,        # largest
        'Continuous': 240,   # slightly smaller
        'Science': 150,      # smaller
        'Fast': 150,         # same as Science
        'Sampled': 240       # smallest
    }

    # Step 3: Plot ALL reach_ids
    for reach_id in all_reach_ids:
        plt.figure(figsize=(25, 10))
        max_Q_all = []

        # --- Build master sequential axis from Continuous (or gauge) ---
        df_gauge = dfs_dict.get('Continuous')  # original gauge df
        df_gauge_reach = df_gauge[(df_gauge['reach_id'] == reach_id) &
                                  (df_gauge['algo'] == 'gauge')].copy() if df_gauge is not None else None

        if df_gauge_reach is not None and not df_gauge_reach.empty:
            df_gauge_reach['time'] = pd.to_datetime(df_gauge_reach['time'])
            df_gauge_reach = df_gauge_reach.sort_values('time')  # ensure ordered
            all_dates = df_gauge_reach['time'].drop_duplicates().reset_index(drop=True)
            master_idx = range(len(all_dates))
            date_to_idx = dict(zip(all_dates, master_idx))
            
            df_gauge_reach['seq_idx'] = df_gauge_reach['time'].map(date_to_idx)
            df_gauge_reach = df_gauge_reach.dropna(subset=['seq_idx'])

            plt.plot(df_gauge_reach['seq_idx'].values, df_gauge_reach['Q'].values,
                        label="GAUGE", color=color_dict.get('gauge', 'black'),
                         alpha=0.5, linewidth=4, zorder=1)
            

            max_Q_all.append(df_gauge_reach['Q'].max())
        else:
            # If no gauge data, build from all available data for this reach_id
            all_dates_list = []
            for label in labels:
                if label == 'gauge' or label not in dfs_dict:
                    continue
                df = dfs_dict[label]
                df_reach_temp = df[(df['reach_id'] == reach_id) & (df['algo'] == algo)].copy()
                if not df_reach_temp.empty:
                    df_reach_temp['time'] = pd.to_datetime(df_reach_temp['time'])
                    all_dates_list.append(df_reach_temp['time'])
            
            if all_dates_list:
                all_dates = pd.concat(all_dates_list).sort_values().drop_duplicates().reset_index(drop=True)
                master_idx = range(len(all_dates))
                date_to_idx = dict(zip(all_dates, master_idx))
            else:
                continue  # Skip if no data at all

        # --- Loop through all other labels in desired order ---
        for label in plot_order:
            if label == 'gauge' or label not in dfs_dict:
                continue  # already plotted or missing

            df = dfs_dict[label]
            color = color_dict.get(label, 'C0')
            marker = marker_styles[labels.index(label) % len(marker_styles)]

            df_reach = df[(df['reach_id'] == reach_id) & (df['algo'] == algo)].copy()
            if df_reach.empty:
                continue

            df_reach['time'] = pd.to_datetime(df_reach['time'])
            df_reach['seq_idx'] = df_reach['time'].map(date_to_idx)
            df_reach = df_reach.dropna(subset=['seq_idx'])

            # Skip Continuous points if all after divide_date
            if label == 'Continuous' and df_reach['time'].min() > divide_date:
                continue
            
            Q = df_reach['Q'].values
            if len(Q) < 10:
                continue
            x_vals = df_reach['seq_idx'].values.astype(int)
            max_Q_all.append(Q.max())

            # Use size from point_sizes
            plt.scatter(x_vals, Q, label=f"{label.upper()} {algo}",
                        alpha=1.0, color=color, marker=marker, s=point_sizes[label])

        # --- Plot vertical divide line + orbit labels ---
        divide_idx = (all_dates < divide_date).sum() if 'all_dates' in locals() else 0
        if divide_idx > 0 and len(max_Q_all) > 0:
            ymax = max(max_Q_all)
            plt.vlines(x=divide_idx, ymin=0, ymax=ymax, colors='black', linewidth=5, linestyle='-')
            plt.text(divide_idx-(divide_idx/2.3), ymax, "FS Orbit",
                     ha='center', va='bottom', fontsize=22, fontweight='bold')
            plt.text(divide_idx + (divide_idx/1.6), ymax, "Science Orbit",
                     ha='center', va='bottom', fontsize=22, fontweight='bold')

        # Labels and formatting
        plt.xlabel(f'Sequential Observations - {reach_id}', fontsize=25)
        plt.ylabel('Discharge (m³/s)', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # if (reach_id) == int(73114000751):
        plt.legend(loc='upper right', fontsize=30)
        
        plt.tight_layout()
        plt.savefig(f'//figs/hydrograph_{reach_id}.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

        

##############
# Priors
##############

def match_priors_to_gauge_data(dfs_gauge, dfs_priors):
    # Create a copy of the input dictionary to avoid modifying the original
    matched_dfs_gauge = dfs_gauge.copy()
    
    for orbit_type, df in matched_dfs_gauge.items():
        # Skip if the DataFrame is empty
        if df.empty:
            continue
        
        # Ensure the month column is added and the time column is datetime
        df['time'] = pd.to_datetime(df['time'])
        df['month'] = df['time'].dt.month
        
        # Get the corresponding prior DataFrame
        prior_df = dfs_priors.get(orbit_type)
        
        if prior_df is not None and not prior_df.empty:
            # Merge the DataFrames on reach_id
            merged_df = df.merge(
                prior_df,
                on='reach_id',
                how='left'
            )
            
            # Function to safely select the correct monthly Q column based on month
            def select_monthly_q(row, q_type):
                month = row['month']
                col_name = f'{q_type}_monthly_q_{month}'
                return row[col_name] if col_name in row.index else np.nan
            
            # Add columns for model and gauge monthly Q based on the month
            try:
                merged_df['matched_model_monthly_q'] = merged_df.apply(lambda row: select_monthly_q(row, 'model'), axis=1)
                merged_df['matched_gauge_monthly_q'] = merged_df.apply(lambda row: select_monthly_q(row, 'gauge'), axis=1)
                
                # Add the matched CAL flag
                merged_df['matched_gauge_CAL_flag'] = merged_df['gauge_CAL_flag'] if 'gauge_CAL_flag' in merged_df.columns else np.nan
            except Exception as e:
                print(f"Error processing {orbit_type}: {e}")
                # If adding columns fails, keep the original DataFrame
                merged_df = df
            
            # Update the DataFrame in the dictionary
            matched_dfs_gauge[orbit_type] = merged_df
    
    return matched_dfs_gauge


def plot_multiple_reaches(dfs_gauge_priors, color_dict, num_reaches=None):
    # Print total number of unique reach_ids for each orbit type
    for orbit_type in ['Fast', 'Science', 'Continuous', 'Sampled']:
        print(f"{orbit_type} unique reach_ids: {dfs_gauge_priors[orbit_type].reach_id.nunique()}")
    
    # Get unique reach_ids with valid monthly model Q
    valid_reaches = {}
    for orbit_type in ['Fast', 'Science', 'Continuous', 'Sampled']:
        df = dfs_gauge_priors[orbit_type]
        
        # Filter for reaches with valid monthly model Q and consensus data
        valid_reaches[orbit_type] = df[
            (df['matched_model_monthly_q'].notna()) & 
            (df['algo'] == 'consensus')]['reach_id'].unique()
        
        print(len(valid_reaches[orbit_type]))
    # Find common reaches across orbit types
    common_reaches = set.intersection(
        *[set(reaches) for reaches in valid_reaches.values()]
    )
    
    # Convert to list and optionally limit the number of reaches
    reach_ids = list(common_reaches)
    if num_reaches is not None:
        reach_ids = reach_ids[:num_reaches]
    
    
    # Create a figure for each reach_id
    for reach_id in reach_ids:
        fig = plot_reach_quad(reach_id, dfs_gauge_priors, color_dict)
        plt.show()

def plot_reach_quad(reach_id, dfs_gauge_priors, color_dict):
    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Reach ID: {reach_id}', fontsize=16)
    
    # Flatten the axes for easy iteration
    axs = axs.flatten()
    
    # List of orbit types
    orbit_types = ['Fast', 'Science', 'Continuous', 'Sampled']
    
    for i, orbit_type in enumerate(orbit_types):
        # Get the dataframe for this orbit type and reach
        df = dfs_gauge_priors[orbit_type]
        df_reach = df[df['reach_id'] == reach_id].copy()  # Use .copy() to avoid warnings
        print(df_reach.matched_gauge_CAL_flag.unique())
        
        
        if df_reach.empty:
            axs[i].text(0.5, 0.5, f'No data for {orbit_type}', 
                        horizontalalignment='center', verticalalignment='center')
            axs[i].set_title(orbit_type)
            continue
        
        # Plot consensus data
        consensus_data = df_reach[df_reach['algo'] == 'consensus'].copy()  # Use .copy()
        if not consensus_data.empty:
            # Convert time to datetime for proper plotting
            consensus_data.loc[:, 'time'] = pd.to_datetime(consensus_data['time'])
            
            # Scatter plot of consensus Q
            axs[i].scatter(consensus_data['time'], 
                           consensus_data['Q'], 
                           color=color_dict['consensus'], 
                           label='Consensus Q')
            
            # Plot monthly model Q as a line that changes over time
            axs[i].plot(consensus_data['time'], 
                        consensus_data['matched_model_monthly_q'], 
                        color=color_dict['consensus'], 
                        linestyle='--', 
                        label='Monthly Model Q')
        
        # Plot gauge_swot_match data
        gauge_data = df_reach[df_reach['algo'] == 'gauge_swot_match'].copy()  # Use .copy()
        if not gauge_data.empty:
            # Convert time to datetime for proper plotting
            gauge_data.loc[:, 'time'] = pd.to_datetime(gauge_data['time'])
            
            axs[i].scatter(gauge_data['time'], 
                           gauge_data['Q'], 
                           color=color_dict['gauge_swot_match'], 
                           label='Gauge SWOT Match Q')
        
        axs[i].set_title(orbit_type)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Discharge (m³/s)')
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].legend()
        
        # Adjust y-axis to show variation
        axs[i].set_ylim(bottom=0)  # Start from 0
    
    plt.tight_layout()
    return fig



def plot_algorithm_metric_cdfs_from_dict(dict_of_dfs, scaling, color_dict, algo_name, dfs_gauge_priors=None):
    """
    Plot CDFs of performance metrics for a specific algorithm across multiple DataFrames.

    Parameters:
    - dict_of_dfs (dict): Dictionary mapping label -> DataFrame
                          (e.g., {'Fast': df1, 'Science': df2, ...})
    - scaling (str): Title suffix for context (e.g., 'All', 'Dry Season', etc.)
    - color_dict (dict): Mapping of labels to plot colors.
    - algo_name (str): The algorithm to filter and plot (default: 'consensus')
    - dfs_gauge_priors (dict): Optional dictionary of DataFrames with prior model data

    Returns:
    - common_reach_ids (set): Set of common reach IDs across Fast, Continuous, and Science
    """
    metricList = ['r', '1-sigma', 'NSE', 'nBIAS'] #, 'KGE', 'RMSE', 'nRMSE', 'rRMSE']

    # Find common reach_ids across Fast, Continuous, and Science only
    runs_to_compare = ['Fast', 'Continuous', 'Science']
    common_reach_ids = None
    
    for label in runs_to_compare:
        if label in dict_of_dfs:
            df = dict_of_dfs[label].copy()
            df['season'] = pd.to_datetime(df['time']).apply(get_season)
            df_algo = df[df['algo'] == algo_name]
            
            reach_ids = set(df_algo['reach_id'].unique())
            
            if common_reach_ids is None:
                common_reach_ids = reach_ids
            else:
                common_reach_ids = common_reach_ids.intersection(reach_ids)
    
    print(f"Number of common reach_ids across Fast, Continuous, and Science: {len(common_reach_ids)}")

    # Calculate gauge_vs_prior metrics for common reach_ids if dfs_gauge_priors provided
    gauge_prior_medians = {}
    if dfs_gauge_priors is not None:
        for metric in metricList:
            orbit_medians = {}
            
            for orbit in ['Fast', 'Continuous', 'Science', 'Sampled']:
                if orbit not in dfs_gauge_priors:
                    continue
                    
                # Prepare the DataFrame
                df = dfs_gauge_priors[orbit].copy()
                df['time'] = pd.to_datetime(df['time'])
                df = df[(df['Q'] > 1) & (df['Q'] < 1e7)]
                df = df[df['matched_gauge_CAL_flag'] == 0.0]
                df = df[df['algo'].isin([algo_name, 'gauge_swot_match'])]
                
                # Filter to common_reach_ids
                if orbit != 'Sampled':
                    df = df[df['reach_id'].isin(common_reach_ids)]
                
                # Add matched_model_monthly_q as an algorithm
                model_df = df.copy()
                model_df['algo'] = 'monthly_priors'
                model_df['Q'] = model_df['matched_model_monthly_q']
                
                # Get gauge data
                gauge_df = df[df['algo'] == 'gauge_swot_match'].copy()
                
                # Combine gauge and model
                full_df = pd.concat([gauge_df, model_df], ignore_index=True)
                
                # Calculate metrics for each reach
                metric_values = []
                reach_ids = full_df['reach_id'].unique()
                
                for reach_id in reach_ids:
                    reach_data = full_df[full_df['reach_id'] == reach_id]
                    
                    gauge_data = reach_data[reach_data['algo'] == 'gauge_swot_match']
                    prior_data = reach_data[reach_data['algo'] == 'monthly_priors']
                    
                    # Merge on time
                    merged_df = pd.merge(
                        gauge_data[['time', 'Q']], 
                        prior_data[['time', 'Q']], 
                        on='time', 
                        suffixes=('_gauge', '_prior')
                    )
                    
                    merged_df = merged_df.dropna()
                    
                    if len(merged_df) < 10:
                        continue
                    
                    X = merged_df['Q_gauge']
                    Y = merged_df['Q_prior']
                    
                    # Calculate metric
                    if metric == 'NSE':
                        val = 1 - (np.sum((X - Y) ** 2) / np.sum((X - X.mean()) ** 2)) if np.sum((X - X.mean()) ** 2) != 0 else np.nan
                    elif metric == 'r':
                        val = pearsonr(X, Y)[0] if len(X) > 1 and len(Y) > 1 else np.nan
                    elif metric == 'nBIAS':
                        val = np.abs((np.sum(X - Y) / len(Y)) / Y.mean()) if Y.mean() != 0 else np.nan
                    elif metric == '1-sigma':
                        norm_res = np.abs((X - Y) / Y.mean()) if Y.mean() != 0 else np.nan
                        val = np.nanpercentile(norm_res, 67) if len(norm_res) > 0 else np.nan
                    else:
                        val = np.nan
                    
                    if not np.isnan(val):
                        metric_values.append(val)
                
                if metric_values:
                    orbit_medians[orbit] = np.median(metric_values)
            
            gauge_prior_medians[metric] = orbit_medians

    # Define marker sizes for each orbit type
    marker_sizes = {
        'Fast': 15,
        'Continuous': 15,
        'Science': 15,
        'Sampled': 15
    }

    # Layout settings
    n_metrics = len(metricList)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 10 * n_rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metricList):
        ax = axes[i]

        for label, df in dict_of_dfs.items():
            if label == 'Continuous-FSO':
                continue
            elif label == 'Continuous-SO':
                continue
                
            df = df.copy()
            df['season'] = pd.to_datetime(df['time']).apply(get_season)
            df_algo = df[df['algo'] == algo_name]
            
            # Filter to only common reach_ids for Fast, Continuous, and Science
            if label in runs_to_compare:
                df_algo = df_algo[df_algo['reach_id'].isin(common_reach_ids)]
            
            x_data = df_algo[metric].dropna()
            if x_data.empty:
                continue
            elif metric == 'nBIAS':
                x_data = x_data.abs()
                
            n_reaches = df_algo[df_algo[metric].notna()]['reach_id'].nunique()
            
            x = np.sort(x_data)
            y = np.arange(1, len(x) + 1) / float(len(x))

            median_val = np.round(np.median(x_data), 2)
            percentile_50 = np.round(np.percentile(x_data, 50), 2)
            
            sns.lineplot(
                x=x, y=y,
                color=color_dict.get(label, None),
                label=f"{label} (n={(n_reaches)}, Median={percentile_50})", linewidth=5.0,
                ax=ax, errorbar=None
            )
            
            # Add gauge_vs_prior median point on the 0.5 line with different sizes
            if dfs_gauge_priors is not None and label in ['Fast', 'Continuous', 'Science', 'Sampled']:
                if metric in gauge_prior_medians and label in gauge_prior_medians[metric]:
                    prior_median = gauge_prior_medians[metric][label]
                    print(label, metric, prior_median)
                    # Special case: Science marker size 25 for nBIAS metric only
                    if metric == 'nBIAS' and label == 'Science':
                        markersize = 30
                    else:
                        markersize = marker_sizes.get(label, 15)
                    
                    ax.plot(prior_median, 0.5, marker='o', 
                           markersize=markersize, 
                           color=color_dict.get(label, None), 
                           markeredgecolor='black', markeredgewidth=2,
                           zorder=10)

        ax.set_xlabel(f'{algo_name} {metric}', fontsize=40)
        ax.set_ylabel('Proportion', fontsize=40)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(y=0.67, color='black', linestyle='--', linewidth=1.5)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=35)

        # Metric-specific x-axis limits
        if metric == 'nBIAS':
            ax.set_xlim(-0.1, 1.25)
            ax.set_xlabel(f'{algo_name} |nBIAS|', fontsize=40)
        elif metric in ['rRMSE', 'nRMSE']:
            ax.set_xlim(0, 2)
        elif metric == 'KGE':
            ax.set_xlim(-1.5, 1.0)
        elif metric == 'NSE':
            ax.set_xlim(-1.0, 1.0)
        elif metric == 'r':
            ax.set_xlim(-0.5, 1)
        elif metric == 'RMSE':
            ax.set_xlim(-0.01, 600)
        elif metric == '1-sigma':
            ax.set_xlim(-0.1, 1.25)

        ax.legend(fontsize=30, frameon=True)

    for j in range(len(axes)):
        if j >= n_metrics:
            axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('//figs/orbit_gauge_CDF_PRIOR_poster.png', bbox_inches='tight', dpi=400)
    plt.show()
    
    return common_reach_ids


def plot_seasonal_log_ratio_by_orbit(dfs_dict, consensus_algo, gauge_algo, monthly_prior_algo, output_dir, plot=True):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp, shapiro, wilcoxon
    import matplotlib.colors as mcolors

    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'SI', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'AR', '9': 'GR'
    }

    def lighten_color(color, amount=0.5):
        """Lighten the given color by mixing it with white."""
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = mcolors.to_rgb(c)
        return tuple(1 - (1 - x) * (1 - amount) for x in c)

    custom_palette = {
        "AF": '#e377c2', "EU": '#2ca02c', "SI": '#17becf',
        "AS": '#d62728', "OC": '#9467bd', "SA": 'gold',
        "NA": '#ff7f0e', "AR": '#1f77b4'
    }

    continent_order = ["AF", "EU", "SI", "AS", "AR", "NA", "SA", "OC"]
    full_season_order = ['4-7', '7-10', '10-1', '1-4']

    def prepare_log_ratio_df(df, consensus_algo, gauge_algo, group_label):
        """Compute seasonal log10(Q_consensus / Q_gauge) by reach."""
        # Filter for specific algorithms
        df_cons = df[df['algo'] == consensus_algo].copy()
        df_gauge = df[df['algo'] == gauge_algo].copy()

        # Check if dataframes are not empty
        if df_cons.empty or df_gauge.empty:
            return pd.DataFrame()

        df_cons['time'] = pd.to_datetime(df_cons['time'])
        df_gauge['time'] = pd.to_datetime(df_gauge['time'])

        # Ensure numeric conversion and positive values
        df_cons['Q'] = pd.to_numeric(df_cons['Q'], errors='coerce')
        df_gauge['Q'] = pd.to_numeric(df_gauge['Q'], errors='coerce')
        
        # If 'continent' is not a column, try to extract it from 'reach_id'
        df['continent'] = df['reach_id'].astype(str).str[0]
        df['continent_name'] = df['continent'].map(continent_map)

        # Debug print sample sizes
        #print(f"Sample sizes for {group_label}:")
        #print(df[df['algo'] == consensus_algo].groupby('continent_name')['reach_id'].nunique())


        merged = pd.merge(
            df_cons[['time', 'reach_id', 'Q']],
            df_gauge[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=('_consensus', '_gauge')
        )
        #print("MERGED1:\n", merged['reach_id'].nunique(), merged[merged.isna().any(axis=1)])

        # Filter for positive, non-NaN values
        merged = merged[
            (merged['Q_consensus'] > 0) & 
            (merged['Q_gauge'] > 0) & 
            pd.notna(merged['Q_consensus']) & 
            pd.notna(merged['Q_gauge'])
        ]
        #print("MERGED2:\n", merged['reach_id'].nunique())

        if merged.empty:
            return pd.DataFrame()

        # Compute log ratio safely
        merged['log_ratio'] = np.log10(merged['Q_consensus']) - np.log10(merged['Q_gauge'])
        
        merged['continent'] = merged['reach_id'].astype(str).str[0]
        merged['continent_name'] = merged['continent'].map(continent_map)
        merged['season'] = merged['time'].apply(get_season_orbits)
        #print("MERGED:\n", merged.groupby('continent_name')['reach_id'].nunique())
        #print("MERGED:\n", merged['reach_id'].nunique())

        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])
        #print("MERGED3:\n", merged.groupby('continent_name')['reach_id'].nunique())


        # Aggregate by reach/season
        reach_medians = (
            merged.groupby(['reach_id', 'season', 'continent_name'])['log_ratio']
            .median().reset_index()
        )
        reach_medians['group'] = group_label
        #print("REACH MEDIANS:", reach_medians.groupby('continent_name')['reach_id'].nunique())

        return reach_medians

    # Loop through orbits
    for orbit_name, df in dfs_dict.items():
        # Determine orbit type and set appropriate figure and font sizes
        is_small_orbit = any(k in orbit_name for k in ['Sampled', 'Fast'])
        if is_small_orbit:
            fig_width = 9
            season_order = [full_season_order[0]]  # Only first season
            
            # Scaled down font sizes
            title_size = 20
            label_size = 18
            tick_size = 16
            legend_size = 14
            guide_size = 14
            star_size = 22
            spread = 6
        else:
            fig_width = 18
            season_order = full_season_order
            
            # Original font sizes
            title_size = 34
            label_size = 32
            tick_size = 28
            legend_size = 24
            guide_size = 24
            star_size = 38

        # Prepare data for this orbit
        df =df[df['matched_gauge_CAL_flag'] == 0.0]

        df_consensus = prepare_log_ratio_df(df, consensus_algo, gauge_algo, 'Consensus')
        df_monthly = prepare_log_ratio_df(df, monthly_prior_algo, gauge_algo, 'Monthly Priors')
        
        # Find overlapping reaches
        overlapping_reaches = set(df_consensus['reach_id']) & set(df_monthly['reach_id'])

        # Filter both DataFrames to include only overlapping reaches
        df_consensus = df_consensus[df_consensus['reach_id'].isin(overlapping_reaches)]
        df_monthly = df_monthly[df_monthly['reach_id'].isin(overlapping_reaches)]
   
        combined = pd.concat([df_consensus, df_monthly], ignore_index=True)
        
        if combined.empty:
            print(f"No valid data found for {orbit_name}.")
            continue

        # Prepare plot
        plt.figure(figsize=(fig_width, 8))

        # Palette with light and dark variants for each continent
        palette = {}
        for cont in continent_order:
            base_color = custom_palette.get(cont, '#333333')
            lighter_color = lighten_color(base_color, amount=0.5)
            palette[f'{cont} Consensus'] = base_color
            palette[f'{cont} Monthly Priors'] = lighter_color

        # Prepare data for plotting
        plot_data = combined.copy()
        
        # Filter for small orbits if necessary
        if is_small_orbit:
            plot_data = plot_data[plot_data['season'] == full_season_order[0]]

        plot_data['display_group'] = plot_data.apply(
            lambda row: f"{row['continent_name']} {row['group']}", 
            axis=1
        )

        # Define consistent hue order
        hue_order = sorted(
            plot_data['display_group'].unique(), 
            key=lambda x: (x.split()[0], x.split()[1])  # Sort by continent, then by group
        )

        # Compute sample sizes for each group
        sample_sizes = plot_data.groupby(['display_group'])['reach_id'].nunique()
        #print(sample_sizes)

        # Significance tests
        group_pvals, group_counts = {}, {}
        for (season, continent, group), g in plot_data.groupby(['season', 'continent_name', 'group']):
            log_vals = g['log_ratio'].dropna()
            if len(log_vals) < 2:
                continue

            #log_vals = modified_z_filter(log_vals)
            if len(log_vals) < 2:
                continue

            group_counts[(season, continent, group)] = len(log_vals)

            try:
                shapiro_p = shapiro(log_vals)[1]
                if shapiro_p > 0.05:
                    _, pval = ttest_1samp(log_vals, popmean=0)
                else:
                    _, pval = wilcoxon(log_vals, alternative='two-sided')
            except Exception:
                pval = np.nan

            group_pvals[(season, continent, group)] = pval

        # Filter out continents with less than 10 samples
        continental_sample_sizes = plot_data.groupby('continent_name')['reach_id'].nunique()
        valid_continents = set(cont for cont, n in continental_sample_sizes.items() if n >= 10)

        # Filter the plot data to include only valid continents
        plot_data = plot_data[plot_data['continent_name'].isin(valid_continents)]

        # Recompute hue order and sample sizes after filtering
        hue_order = sorted(
            plot_data['display_group'].unique(), 
            key=lambda x: (x.split()[0], x.split()[1])  # Sort by continent, then by group
        )
        sample_sizes = plot_data.groupby(['display_group'])['reach_id'].nunique()

        # Create boxplot with specified hue order
        ax = sns.boxplot(
            data=plot_data,
            x='season',
            y='log_ratio',
            hue='display_group',
            order=season_order,
            hue_order=hue_order,
            palette=palette
        )

        # Modify legend labels to include sample sizes
        handles, labels = ax.get_legend_handles_labels()
        new_legend_labels = [f"{label} (n={sample_sizes.get(label, 0)})" for label in labels]
        ax.legend(handles, new_legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_size)

        # Add significance stars
        season_list = season_order
        continent_list = [g.split()[0] for g in hue_order]

        # Track which seasons have significant results
        significant_results = {}

        for (season, continent, group), group_df in plot_data.groupby(['season', 'continent_name', 'group']):
            if continent not in continent_list or season not in season_list:
                continue

            season_idx = season_list.index(season)
            continent_idx = continent_list.index(continent)

            pval = group_pvals.get((season, continent, group), None)
            if pval is not None and pval < 0.05:
                # Use the exact color from the palette
                label = f"{continent} {group}"
                star_color = palette.get(label, 'black')

                # Initialize season tracking if not exists
                if season not in significant_results:
                    significant_results[season] = []

                significant_results[season].append({
                    'continent': continent,
                    'group': group,
                    'color': star_color,
                    'label': label
                })

        # Place stars after collecting all significant results
        for season, results in significant_results.items():
            season_idx = season_list.index(season)

            # Sort results to ensure consistent positioning
            results.sort(key=lambda x: continent_list.index(x['continent']))

            for i, result in enumerate(results):
                # Use the continent's index to determine horizontal position
                continent_idx = continent_list.index(result['continent'])
                spread = 0.55  # Adjust this to match boxplot width
                offset = -0.35 + (i * 0.1)  # Base offset
                x_pos = season_idx + offset + (continent_idx * spread / len(continent_list) * 1.4)

                # Add star with vertically oriented text
                ax.text(
                    x_pos, ax.get_ylim()[1] * 0.45, '*', 
                    ha='center', va='bottom', 
                    fontsize=star_size, 
                    color=result['color'], 
                    fontweight='bold'
                )

        # Reference lines
        ax.axhline(0, color='k', linestyle='--', linewidth=2)
        for val, label in zip([np.log10(2), np.log10(3), np.log10(5),
                               -np.log10(2), -np.log10(3), -np.log10(5)],
                              ['2x', '3x', '5x', '2x', '3x', '5x']):
            ax.axhline(val, linestyle='--', color='gray', linewidth=1, alpha=0.6)

        # Title and formatting
        plt.title(f"{orbit_name}", fontsize=title_size)
        plt.xlabel("Season", fontsize=label_size)
        plt.ylabel("log10(Discharge / Gauge)", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Seasonal_LogRatio_{orbit_name}_{consensus_algo}_{monthly_prior_algo}_{gauge_algo}_.png"), dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()
        
        

def plot_seasonal_log_ratio_by_continent(
    dfs_dict, consensus_algo, gauge_algo, monthly_prior_algo, output_dir, plot=True
):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp, shapiro, wilcoxon
    import matplotlib.colors as mcolors

    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'Siberia', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'Arctic', '9': 'GR'
    }

    def lighten_color(color, amount=0.5):
        """Lighten the given color by mixing it with white."""
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = mcolors.to_rgb(c)
        return tuple(1 - (1 - x) * (1 - amount) for x in c)

    custom_palette = {
        "AF": '#e377c2', "EU": '#2ca02c', "Siberia": '#17becf',
        "AS": '#d62728', "OC": '#9467bd', "SA": 'gold',
        "NA": '#ff7f0e', "Arctic": '#1f77b4'
    }

    continent_order = ["AF", "EU", "Siberia", "AS", "Arctic", "NA", "SA", "OC"]
    full_season_order = ['4-7', '7-10', '10-1', '1-4']

    def prepare_log_ratio_df(orbit_dfs, consensus_algo, gauge_algo, group_label):
        """Compute seasonal log10(Q_consensus / Q_gauge) by reach across all orbits."""
        combined_merged = []

        for orbit, df in orbit_dfs.items():
            # Filter for specific algorithms
            df_cons = df[df['algo'] == consensus_algo].copy()
            df_gauge = df[df['algo'] == gauge_algo].copy()

            # Check if dataframes are not empty
            if df_cons.empty or df_gauge.empty:
                continue

            df_cons['time'] = pd.to_datetime(df_cons['time'])
            df_gauge['time'] = pd.to_datetime(df_gauge['time'])

            # Ensure numeric conversion and positive values
            df_cons['Q'] = pd.to_numeric(df_cons['Q'], errors='coerce')
            df_gauge['Q'] = pd.to_numeric(df_gauge['Q'], errors='coerce')

            merged = pd.merge(
                df_cons[['time', 'reach_id', 'Q']],
                df_gauge[['time', 'reach_id', 'Q']],
                on=['time', 'reach_id'],
                suffixes=('_consensus', '_gauge')
            )

            # Filter for positive, non-NaN values
            merged = merged[
                (merged['Q_consensus'] > 0) & 
                (merged['Q_gauge'] > 0) & 
                pd.notna(merged['Q_consensus']) & 
                pd.notna(merged['Q_gauge'])
            ]

            if merged.empty:
                continue

            # Compute log ratio safely
            merged['log_ratio'] = np.log10(merged['Q_consensus']) - np.log10(merged['Q_gauge'])
            
            merged['continent'] = merged['reach_id'].astype(str).str[0]
            merged['continent_name'] = merged['continent'].map(continent_map)
            merged['season'] = merged['time'].apply(get_season_orbits)
            
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])

            combined_merged.append(merged)

        if not combined_merged:
            return pd.DataFrame()

        # Concatenate data from all orbits
        full_merged = pd.concat(combined_merged, ignore_index=True)

        # Aggregate by reach/season
        reach_medians = (
            full_merged.groupby(['reach_id', 'season', 'continent_name'])['log_ratio']
            .median().reset_index()
        )
        reach_medians['group'] = group_label
        return reach_medians

    def significance_tests(df):
        """Perform per-season, per-continent significance test for each group independently."""
        group_pvals, group_counts = {}, {}

        for (season, cont, group), g in df.groupby(['season', 'continent_name', 'group']):
            vals = g['log_ratio'].dropna()
            if len(vals) < 2 or cont == 'GR':
                continue

            # Optional outlier filtering
            vals = modified_z_filter(vals)
            if len(vals) < 2:
                continue

            group_counts[(season, cont, group)] = len(vals)

            try:
                # Normality check
                if shapiro(vals)[1] > 0.05:
                    _, p = ttest_1samp(vals, popmean=0)
                else:
                    _, p = wilcoxon(vals, alternative='two-sided')
            except Exception:
                p = np.nan

            group_pvals[(season, cont, group)] = p

        return group_pvals, group_counts

    # Prepare data
    df_consensus = prepare_log_ratio_df(dfs_dict, consensus_algo, gauge_algo, 'Consensus')
    df_monthly = prepare_log_ratio_df(dfs_dict, monthly_prior_algo, gauge_algo, 'Monthly Priors')

    combined = pd.concat([df_consensus, df_monthly], ignore_index=True)
    
    if combined.empty:
        print("No valid data found for comparison.")
        return

    # Loop over continents
    for cont in continent_order:
        df_cont = combined[combined['continent_name'] == cont].copy()
        if df_cont.empty:
            continue

        group_pvals, group_counts = significance_tests(df_cont)

        # Drop sparse groups
        df_cont = df_cont[df_cont.apply(
            lambda row: group_counts.get((row['season'], row['continent_name'], row['group']), 0) > 10,
            axis=1
        )]
        if df_cont.empty:
            continue

        season_order = [s for s in full_season_order if s in df_cont['season'].unique()]
        if not season_order:
            continue

        sample_sizes = df_cont.groupby(['group'])['reach_id'].nunique().to_dict()
        print(f"{cont} sample sizes:", sample_sizes)

        # Filter out groups with <10 reaches
        valid_groups = [g for g, n in sample_sizes.items() if n >= 10]
        df_cont = df_cont[df_cont['group'].isin(valid_groups)]

        # Skip if nothing valid remains
        if df_cont.empty:
            print(f"Skipping {cont} — no group with ≥10 reaches.")
            continue

        if plot:
            base_color = custom_palette.get(cont, '#333333')
            lighter_color = lighten_color(base_color, amount=0.5)
            palette = {'Consensus': base_color, 'Monthly Priors': lighter_color}

            plt.figure(figsize=(12, 10))
            ax = sns.boxplot(
                data=df_cont,
                x='season',
                y='log_ratio',
                hue='group',
                order=season_order,
                palette=palette
            )

            # Reference lines
            plt.axhline(0, color='k', linestyle='--', linewidth=2)
            for val, label in zip([np.log10(2), np.log10(3), np.log10(5),
                                   -np.log10(2), -np.log10(3), -np.log10(5)],
                                  ['2x', '3x', '5x', '2x', '3x', '5x']):
                plt.axhline(val, linestyle='--', color='gray', linewidth=1, alpha=0.6)
                plt.text(ax.get_xlim()[1] - 0.2, val, label, color='gray', fontsize=25, va='bottom')

            # Significance stars
            for season in season_order:
                for group in ['Consensus', 'Monthly Priors']:
                    n = group_counts.get((season, cont, group), 0)
                    if n > 10:
                        p = group_pvals.get((season, cont, group), None)
                        if p is not None and p < 0.05:
                            xpos = season_order.index(season)
                            star_color = palette[group]
                            ax.text(
                                xpos - 0.1 if group == 'Consensus' else xpos + 0.1,
                                ax.get_ylim()[1] * 0.85,
                                '*',
                                ha='center',
                                va='bottom',
                                fontsize=30,
                                color=star_color,
                                fontweight='bold'
                            )

            # Legend with sample sizes
            handles, labels = ax.get_legend_handles_labels()
            new_labels = []
            for label in labels:
                n_val = sample_sizes.get(label, 0)
                new_labels.append(f"{label} (n={n_val})")

            ax.legend(
                handles,
                new_labels,
                title='Group',
                fontsize=24,
                title_fontsize=28,
                loc='lower left'
            )

            plt.title(f"{cont}", fontsize=34)
            plt.xlabel("Season", fontsize=32)
            plt.ylabel("Discharge Ratio (log)", fontsize=32)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.tight_layout()

            outpath = os.path.join(output_dir, f"Seasonal_LogRatio_{cont}_Consensus_MonthlyPriors.png")
            plt.savefig(outpath, dpi=500)
            plt.show()
            plt.close()
            print(f"Saved {outpath}")
            

# Helper function to filter outliers using modified Z-score
def modified_z_filter(series, threshold=3.5):
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return series
    mzs = 0.6745 * (series - median) / mad
    return series[np.abs(mzs) <= threshold]


def plot_grouped_boxplot_differences(dict_of_dfs, algo_name, color_dict=None, plot=True):
    metricList = ['r', 'NSE', 'nBIAS', '1-sigma']
    metricLabels = {'r': 'Pearson r', 'NSE': 'NSE', 'nBIAS': '|nBIAS|', '1-sigma': 'sigE'}
    selected_comparisons = [
        ('Fast', 'Continuous'),
        ('Science', 'Continuous'),
        ('Fast', 'Sampled'),
    ]

    hue_colors = {
        'Pearson r': '#1f77b4',
        'NSE': '#ff7f0e',
        '|nBIAS|': '#2ca02c',
        'sigE': '#d62728',
        'log(Q1/Q2)': '#9467bd'
    }
    hue_colors = {
            'Pearson r': '#f9f9f9',   # very light gray
            'NSE': '#d9d9d9',         # light gray
            '|nBIAS|': '#a6a6a6',     # medium gray
            'sigE': '#737373',        # dark gray
            'log(Q1/Q2)': '#4D4D4D'#'#6D676E'   # darker gray
        }
    full_diff_records = []

    # --- Metric differences: only one observation per reach ---
    for run1, run2 in selected_comparisons:
        df1 = dict_of_dfs[run1]
        df2 = dict_of_dfs[run2]

        df1 = df1[df1['algo'] == algo_name].drop_duplicates('reach_id')
        df2 = df2[df2['algo'] == algo_name].drop_duplicates('reach_id')

        merged = pd.merge(
            df1[['reach_id'] + metricList],
            df2[['reach_id'] + metricList],
            on='reach_id',
            suffixes=(f'_{run1}', f'_{run2}')
        )

        for metric in metricList:
            col1, col2 = f'{metric}_{run1}', f'{metric}_{run2}'
            if metric == 'nBIAS':
                diffs = merged[col1].abs() - merged[col2].abs()
            else:
                diffs = merged[col1] - merged[col2]

            diffs = diffs.dropna()
            diffs_filtered = modified_z_filter(diffs)

            pval = np.nan
            try:
                if len(diffs_filtered) > 1:
                    if shapiro(diffs_filtered)[1] > 0.05:
                        _, pval = ttest_1samp(diffs_filtered, 0)
                    else:
                        _, pval = wilcoxon(diffs_filtered)
            except Exception:
                pass

            for val in diffs_filtered:
                full_diff_records.append({
                    'Comparison': f'{run1} - {run2}',
                    'Metric': metricLabels[metric],
                    'Difference': val,
                    'pval': pval
                })

    # --- Q log-ratio differences: keep all observations per reach ---
    for run1, run2 in selected_comparisons:
        dfq1 = dict_of_dfs[run1]
        dfq2 = dict_of_dfs[run2]

        dfq1 = dfq1[dfq1['algo'] == algo_name].copy()
        dfq2 = dfq2[dfq2['algo'] == algo_name].copy()

        dfq1['time'] = pd.to_datetime(dfq1['time'], errors='coerce')
        dfq2['time'] = pd.to_datetime(dfq2['time'], errors='coerce')

        merged_q = pd.merge(
            dfq1[['time', 'reach_id', 'Q']],
            dfq2[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=(f"_{run1}", f"_{run2}")
        )

        merged_q[f"Q_{run1}"] = pd.to_numeric(merged_q[f"Q_{run1}"], errors="coerce")
        merged_q[f"Q_{run2}"] = pd.to_numeric(merged_q[f"Q_{run2}"], errors="coerce")


        merged_q = merged_q.dropna(subset=[f"Q_{run1}", f"Q_{run2}"])
        merged_q = merged_q[(merged_q[f"Q_{run1}"] > 0) & (merged_q[f"Q_{run2}"] > 0)]
        merged_q['log_ratio'] = np.log10(merged_q[f"Q_{run1}"]) - np.log10(merged_q[f"Q_{run2}"])

        log_ratios = merged_q['log_ratio'].dropna()
        log_ratios_filtered = modified_z_filter(log_ratios)

        try:
            if len(log_ratios_filtered) > 3:
                if shapiro(log_ratios_filtered)[1] > 0.05:
                    pval = ttest_1samp(log_ratios_filtered, 0).pvalue
                else:
                    pval = wilcoxon(log_ratios_filtered).pvalue
            else:
                pval = np.nan
        except Exception:
            pval = np.nan

        for val in log_ratios_filtered:
            full_diff_records.append({
                'Metric': 'log(Q1/Q2)',
                'Difference': val,
                'Comparison': f"{run1} - {run2}",
                'pval': pval
            })

    # --- Build DataFrame ---
    all_df = pd.DataFrame(full_diff_records)
    comparison_order = [f'{r1} - {r2}' for r1, r2 in selected_comparisons]
    all_df['Comparison'] = pd.Categorical(all_df['Comparison'], categories=comparison_order, ordered=True)

    # Plot
    if plot:
        def star(p): return '*' if (not pd.isna(p) and p < 0.05) else ''
        plt.figure(figsize=(16, 8))
        ax = sns.boxplot(
            data=all_df, x='Comparison', y='Difference', hue='Metric',
            palette=hue_colors, order=comparison_order
        )
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Metric Differences by Orbit: {algo_name.capitalize()}', fontsize=26)
        plt.ylabel('Difference', fontsize=26)
        plt.xlabel('Orbit Comparison', fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)

        for i, (comparison, group) in enumerate(all_df.groupby('Comparison')):
            for j, metric in enumerate(group['Metric'].unique()):
                sub = group[group['Metric'] == metric]
                if not sub.empty:
                    median_val = sub['Difference'].median()
                    xpos = i - 0.3 + 0.16 * j
                    color = hue_colors.get(metric, 'black')
                    ax.text(xpos, median_val + 1, star(sub['pval'].iloc[0]),
                            ha='center', va='bottom', fontsize=32,
                            color='black', weight='bold')
                    ax.text(xpos-0.02, -1.3, f'n={len(sub)}', ha='center', fontsize=14)

        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=24)
        plt.savefig('//figs/gauge_cons_metric_comparison.png', dpi=350)
        plt.show()

    return all_df


def plot_grouped_boxplot_runs_by_reach(dfs_dict, consensus_algo, color_dict=None):

    valid_pairs = [
        ('Fast', 'Continuous'),
        ('Science', 'Continuous'),
        ('Fast', 'Sampled')
    ]

    all_records = []

    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'SI', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'AR', '9': 'GR'
    }

    custom_palette = {
        "AF": '#e377c2', "EU": '#2ca02c', "SI": '#17becf',
        "AS": '#d62728', "NA": '#ff7f0e', "AR": '#1f77b4', "GR": 'gray'
    }

    for name1, name2 in valid_pairs:
        if name1 not in dfs_dict or name2 not in dfs_dict:
            continue

        df1 = dfs_dict[name1]
        df2 = dfs_dict[name2]

        df1_c = df1[df1['algo'] == consensus_algo].copy()
        df2_c = df2[df2['algo'] == consensus_algo].copy()

        df1_c['time'] = pd.to_datetime(df1_c['time'], errors='coerce')
        df2_c['time'] = pd.to_datetime(df2_c['time'], errors='coerce')

        merged = pd.merge(
            df1_c[['time', 'reach_id', 'Q']],
            df2_c[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=(f"_{name1}", f"_{name2}")
        ).dropna()

        #  Assure numeric Q values
        merged[f"Q_{name1}"] = pd.to_numeric(merged[f"Q_{name1}"], errors='coerce')
        merged[f"Q_{name2}"] = pd.to_numeric(merged[f"Q_{name2}"], errors='coerce')
        merged = merged.dropna()

        merged = merged[(merged[f"Q_{name1}"] > 0) & (merged[f"Q_{name2}"] > 0)]
        merged['log_ratio'] = (
            np.log10(merged[f"Q_{name1}"].to_numpy()) -
            np.log10(merged[f"Q_{name2}"].to_numpy())
        )
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])

        merged['continent_name'] = merged['reach_id'].astype(str).str[0].map(continent_map)
        merged = merged[~merged['continent_name'].isin(['SA', 'OC', 'AF'])]

        if merged.empty:
            continue

        reach_grp = merged.groupby(['reach_id', 'continent_name'])['log_ratio'].median().reset_index()

        reach_grp['run_label'] = f"{name1}/{name2}"
        all_records.append(reach_grp)

    if not all_records:
        print("No data after processing.")
        return

    plot_df = pd.concat(all_records)
    print(plot_df.groupby('continent_name')['reach_id'].nunique())

   # ---- Stats ----
    group_pvals = {}
    group_counts = {}
    n_by_continent = {}

    for (run_label, continent_name), group_df in plot_df.groupby(['run_label', 'continent_name']):
        vals = group_df['log_ratio']

        # number of unique reaches for this run + continent
        n = group_df['reach_id'].nunique()
        group_counts[(run_label, continent_name)] = n
        if n < 10:
            print(run_label, continent_name, n)
            continue

        # accumulate unique reach IDs per continent for legend
        if continent_name not in n_by_continent:
            n_by_continent[continent_name] = set()
        n_by_continent[continent_name].update(group_df['reach_id'].unique())

        # statistics
        if n > 1:
            shapiro_p = shapiro(vals)[1]
            if shapiro_p > 0.05:
                _, p = ttest_1samp(vals, 0)
            else:
                _, p = wilcoxon(vals)
        else:
            p = np.nan
        group_pvals[(run_label, continent_name)] = p

    # Convert sets to counts for legend
    n_by_continent = {k: len(v) for k, v in n_by_continent.items()}

    # ---- Plot ----
    plt.figure(figsize=(16, 7))
    ax = sns.boxplot(
        data=plot_df,
        x='run_label',
        y='log_ratio',
        hue='continent_name',
        palette=custom_palette,
        showfliers=False
    )
    
    
    # Log10 difference lines at 2x, 3x, 5x and their inverses (0.5x, 0.33x, 0.2x)
    x_pos = len(list(plot_df['run_label'].unique())) - 0.4
    for val, label in zip([0, np.log10(2), np.log10(3),
                           -np.log10(2), -np.log10(3)],
                          ['1x', '2x', '3x', '2x', '3x']):
        plt.axhline(val, linestyle='--', color='gray', linewidth=1.5, alpha=0.5)
        plt.text(x_pos-0.09, val, label, color='black', fontsize=22, va='bottom')
        
        
    plt.axhline(0, color='k', linestyle='--', linewidth=1.5)
    plt.title("Discharge Magnitude Differences Across Runs", fontsize=28)
    plt.xlabel("Run Comparison", fontsize=26)
    plt.ylabel("log10(Q_run₁ / Q_run₂)", fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.ylim([-0.5, 0.5])

    # Asterisks + n below boxes
    grouped = plot_df.groupby(['run_label', 'continent_name'])
    run_labels = [f"{a}/{b}" for a, b in valid_pairs if a in dfs_dict and b in dfs_dict]    
    continent_list = [c for c in plot_df['continent_name'].unique().tolist() if c not in ['SA', 'OC']]
    spread = 0.3
    offset = -0.3

    for (run_label, continent), group_df in grouped:
        run_idx = run_labels.index(run_label)
        continent_idx = continent_list.index(continent)
        x_pos = run_idx + offset + continent_idx * spread

        pval = group_pvals.get((run_label, continent), None)
        n = group_df['reach_id'].nunique()

        if pval is not None and pval < 0.05:
            star_color = custom_palette.get(continent, 'black')
            ax.text(x_pos, 0.3, '*', ha='center', va='bottom', fontsize=25,
                    color=star_color, weight='bold')

        #ax.text(x_pos, -0.95, f"n={n}", ha='center', va='top', fontsize=11)

    # custom legend with n values
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{lbl} (n={n_by_continent.get(lbl, 0)})" for lbl in labels]
    ax.legend(handles, new_labels, title="Continent", fontsize=18, title_fontsize=20, loc='lower left')

    plt.tight_layout()
    plt.savefig('//figs/runQcomparions_gauge.png', dpi=350)
    plt.show()


######################
# Quantile performance
#######################

def plot_nd_scatter_by_quantile_per_df(dfs_dict, consensus_algo, gauge_algo):
    """
    Compute ND between consensus and gauge discharge per DataFrame in dfs_dict,
    at 0.1, 0.5, 0.9 quantiles, then plot:
      - 1:1 scatterplots by quantile
      - Histogram of ND per quantile
    Prints mean ± SD of ND and calculates MAPE and rRMSE per quantile.
    """
    qs = [0.1, 0.5, 0.9]
    all_results = {}

    for df_name, df in dfs_dict.items():
        if df_name in ['Continuous-SO', 'Continuous-FSO']:
            continue
        print(f"\n=== Processing DataFrame: {df_name} ===")
        
        # Keep only consensus + gauge
        df_c = df[df['algo'] == consensus_algo].copy()
        df_g = df[df['algo'] == gauge_algo].copy()

        # Merge on time + reach
        merged = pd.merge(
            df_c[['time', 'reach_id', 'Q']],
            df_g[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=('_cons', '_gauge')
        )

        # Keep positive flows
        merged = merged[(merged['Q_cons'] > 0) & (merged['Q_gauge'] > 0)]

        # Compute quantiles per reach
        results = []
        for reach, group in merged.groupby('reach_id'):
            for q in qs:
                q_cons = group['Q_cons'].quantile(q)
                q_gauge = group['Q_gauge'].quantile(q)
                if q_gauge > 0:
                    nd = np.log10(q_cons) - np.log10(q_gauge)
                    nd = np.nan if np.isinf(nd) else nd
                    results.append({
                        'reach_id': reach,
                        'quantile': q,
                        'Q_cons': q_cons,
                        'Q_gauge': q_gauge,
                        'ND': nd
                    })

        df_q = pd.DataFrame(results)

        # Add continent
        df_q['continent'] = df_q['reach_id'].astype(str).str[0]
        continent_map = {
            '1': 'AF', '2': 'EU', '3': 'SI', '4': 'AS',
            '5': 'OC', '6': 'SA', '7': 'NA', '8': 'AR', '9': 'GR'
        }
        df_q['continent_name'] = df_q['continent'].map(continent_map)
        
        #Filter for low reach numbers
        #df_q = df_q[~df_q['continent_name'].isin(['OC', 'SA'])]

        # Sample sizes & palette
        sample_sizes_reach = df_q.groupby('continent_name')['reach_id'].nunique().to_dict()
        sample_sizes_Q = df_q.groupby('continent_name').size().to_dict()

        # ---- Build custom palette with sample sizes ----
        base_colors = {
            'EU': '#2ca02c',
            'NA': '#ff7f0e',
            'AR': '#1f77b4',
            'SA': 'palegoldenrod',
            'OC': 'purple',
            'AF': 'brown',
            'SI': 'pink',
            'AS': 'gray',
            'GR': 'teal'
        }

        color_dict = {
            f'{cont} (n_gauge={sample_sizes_reach.get(cont, 0)}, n_Q={sample_sizes_Q.get(cont, 0)})':
                base_colors.get(cont, 'black')
            for cont in base_colors
        }

        df_q['continent_label'] = df_q['continent_name'].apply(
            lambda c: f'{c} (n_gauge={sample_sizes_reach.get(c, 0)}, n_Q={sample_sizes_Q.get(c, 0)})'
        )

        # Metrics per quantile & histograms
        metrics = {}
        fig_hist, axes_hist = plt.subplots(1, 3, figsize=(40, 15))
        for ax, q in zip(axes_hist, qs):
            sub = df_q[df_q['quantile'] == q]
            
            # Apply modified z-filter
            nd_filtered = modified_z_filter(sub['ND'])
                        
            rel_errors = nd_filtered
            mape = np.mean(np.abs(rel_errors)) * 100
            rrmse = (np.sqrt(np.mean(rel_errors**2)))  * 100


            mean_nd = np.mean(rel_errors)
            sd_nd = np.std(rel_errors)
            
            metrics[q] = {'ND_mean': mean_nd, 'ND_sd': sd_nd, 'MAPE (%)': mape, 'rRMSE (%)': rrmse}

            print(f"Quantile {q}: ND mean = {mean_nd:.4f}, SD = {sd_nd:.4f}, MAPE = {mape:.2f}%, rRMSE = {rrmse:.2f}%")

            # Histogram
            #plt.figure(figsize=(8,5))
            sns.histplot(rel_errors, bins=50, kde=True, color='skyblue', ax=ax)
            ax.set_title(f'{df_name} - ND Histogram - Quantile {q}', fontsize=25)
            ax.set_xlabel('ND = (Q_cons - Q_gauge)/Q_gauge', fontsize=20)
            ax.set_ylabel('Count', fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(axis='both', labelsize=18)

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.show()

        # Scatterplots
        fig, axes = plt.subplots(1, 3, figsize=(40, 18))
        for ax, q in zip(axes, qs):
            sub = df_q[df_q['quantile'] == q]
            
             # Apply modified z-filter
            sub_filtered = sub.loc[modified_z_filter(sub['ND']).index]

            sns.scatterplot(
                data=sub_filtered,
                x='Q_gauge',
                y='Q_cons',
                hue='continent_label',
                palette=color_dict,
                ax=ax,
                s=150,
                edgecolor='black',
                alpha=0.8
            )
            if not sub.empty:
                max_val = max(sub_filtered['Q_gauge'].max(), sub_filtered['Q_cons'].max())
                ax.plot([0, max_val], [0, max_val], '--k', lw=1.5)  # 1:1 line
            ax.set_title(f'Quantile {q}', fontsize=44)
            ax.set_xlabel(f'{df_name} Gauge Q (m³/s)', fontsize=50)
            ax.set_ylabel(f'{df_name} Consensus Q (m³/s)', fontsize=50)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', labelsize=45)
            ax.legend(title='Continent', title_fontsize=32, fontsize=29, loc='upper left')
            
            # Add info box in lower-right corner
            nd_mean = metrics[q]['ND_mean']
            mape = metrics[q]['MAPE (%)']
            info_text = f"Mean ND: {nd_mean:.3f}\nMAPE: {mape:.2f}%"
            ax.text(
                0.95, 0.05, info_text,
                transform=ax.transAxes,
                fontsize=42,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor='black')
            )
            
            
            
        plt.tight_layout()
        plt.savefig(f'//figs/gaugeQuantile_{df_name}.png')
        plt.show()

        all_results[df_name] = {'df_q': df_q, 'metrics': metrics}

    return all_results


###########
## Algo performance comparison
############
def sign_test(values, mu0=0):
    """Simple two-sided sign test."""
    diffs = np.array(values) - mu0
    diffs = diffs[diffs != 0]  # drop zeros
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan
    pluses = np.sum(diffs > 0)
    res = binomtest(pluses, n, p=0.5, alternative='two-sided')
    return pluses, res.pvalue

def plot_selected_algorithm_differences_grouped_by_continent(dict_of_dfs, algo_name, color_dict=None, plot=False):

    metricList = ['r', 'NSE', 'nBIAS', '1-sigma']
    metricLabels = {'r': 'Pearson r', 'NSE': 'NSE', 'nBIAS': '|nBIAS|', '1-sigma': 'sigE'}

    selected_comparisons = [
        ('Fast', 'Continuous'),
        ('Science', 'Continuous'),
        ('Fast', 'Sampled'),
    ]

    for run1, run2 in selected_comparisons:
        df1 = dict_of_dfs[run1]
        df2 = dict_of_dfs[run2]

        df1 = df1[df1['algo'] == algo_name].copy()
        df2 = df2[df2['algo'] == algo_name].copy()

        df1 = df1.drop_duplicates('reach_id')
        df2 = df2.drop_duplicates('reach_id')

        merged = pd.merge(
            df1[['reach_id'] + metricList + ['Q']],
            df2[['reach_id'] + metricList + ['Q']],
            on='reach_id',
            suffixes=(f'_{run1}', f'_{run2}')
        )
        
        merged['continent'] = merged['reach_id'].astype(str).str[0]
        continent_map = {
            '1': 'AF', '2': 'EU', '3': 'Northern AS', '4': 'Southern AS',
            '5': 'OC', '6': 'SA', '7': 'Southern NA', '8': 'Northern NA', '9': 'GR'
        }
        merged['Continent'] = merged['continent'].map(continent_map)
        merged = merged.dropna(subset=['Continent'])
        merged = merged[merged['Continent'] != 'GR']
                        
        diff_records = []
        sig_labels = {}

        for metric in metricList:
            col1 = f'{metric}_{run1}'
            col2 = f'{metric}_{run2}'

            diffs = merged[col1].abs() - merged[col2].abs() if metric == 'nBIAS' else merged[col1] - merged[col2]
            diffs = diffs.dropna()

            # Q1 = diffs.quantile(0.25)
            # Q3 = diffs.quantile(0.75)
            # IQR = Q3 - Q1
            # diffs = diffs[(diffs >= Q1 - 1.5 * IQR) & (diffs <= Q3 + 1.5 * IQR)]
            diffs = modified_z_filter(diffs)

            for idx in diffs.index:
                diff_records.append({
                    'Metric': metricLabels[metric],
                    'Difference': diffs.loc[idx],
                    'Continent': merged.loc[idx, 'Continent']
                })

        # --- Add Q log-ratio difference as another "metric" ---
        dfq1 = dict_of_dfs[run1]
        dfq2 = dict_of_dfs[run2]

        dfq1 = dfq1[dfq1['algo'] == algo_name].copy()
        dfq2 = dfq2[dfq2['algo'] == algo_name].copy()

        dfq1['time'] = pd.to_datetime(dfq1['time'], errors='coerce')
        dfq2['time'] = pd.to_datetime(dfq2['time'], errors='coerce')

        merged_q = pd.merge(
            dfq1[['time', 'reach_id', 'Q']],
            dfq2[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=(f"_{run1}", f"_{run2}")
        )

        # Filter to positive Q values only (and ensure numeric conversion)
        merged_q[f"Q_{run1}"] = pd.to_numeric(merged_q[f"Q_{run1}"], errors='coerce')
        merged_q[f"Q_{run2}"] = pd.to_numeric(merged_q[f"Q_{run2}"], errors='coerce')

        # Drop rows with invalid or zero/non-positive Q
        merged_q = merged_q.dropna(subset=[f"Q_{run1}", f"Q_{run2}"])
        merged_q = merged_q[(merged_q[f"Q_{run1}"] > 0) & (merged_q[f"Q_{run2}"] > 0)]

        # Now compute log-ratio
        merged_q['log_ratio'] = np.log10(merged_q[f"Q_{run1}"]) - np.log10(merged_q[f"Q_{run2}"])
        merged_q = merged_q.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])

        # Map continent using leading digit of reach_id
        merged_q['continent'] = merged_q['reach_id'].astype(str).str[0]
        merged_q['Continent'] = merged_q['continent'].map(continent_map)
        merged_q = merged_q.dropna(subset=['Continent'])

        # Apply IQR filtering
        for continent, group in merged_q.groupby('Continent'):
            log_ratios = group['log_ratio'].dropna()
            log_ratios_filtered = modified_z_filter(log_ratios)

            for idx in log_ratios_filtered.index:
                diff_records.append({
                    'Metric': 'log(Q1/Q2)',
                    'Difference': log_ratios_filtered.loc[idx],
                    'Continent': continent
                })

        diff_df = pd.DataFrame(diff_records)

               # Significance testing
        for metric in diff_df['Metric'].unique():
            for continent in diff_df['Continent'].unique():
                subset = diff_df[(diff_df['Metric'] == metric) & (diff_df['Continent'] == continent)]
                values = subset['Difference'].dropna()
                if len(values) > 1:
                    try:
                        if shapiro(values)[1] > 0.05:
                            # Data ~ normal → one-sample t-test
                            _, pval = ttest_1samp(values, 0)
                        else:
                            # Try Wilcoxon
                            with warnings.catch_warnings(record=True) as wlist:
                                warnings.simplefilter("always")
                                _, pval = wilcoxon(values - 0)
                                # If SciPy warns about zeros → fallback to sign test
                                if any("zeros" in str(w.message) for w in wlist):
                                    _, pval = sign_test(values, mu0=0)
                        sig_labels[(metric, continent)] = pval
                    except Exception:
                        # Fallback if Wilcoxon completely fails (e.g. all zeros)
                        _, pval = sign_test(values, mu0=0)
                        sig_labels[(metric, continent)] = pval


        # Count per-metric (reach-based) and Q (time-based) samples by continent
        sample_sizes_reach = merged.groupby("Continent")["reach_id"].nunique().to_dict()
        sample_sizes_Q = merged_q.groupby("Continent")["log_ratio"].count().to_dict()

        # Apply to diff_df legend labels
        diff_df["Continent (n)"] = diff_df["Continent"].apply(
            lambda x: f"{x} (n_Metric={sample_sizes_reach.get(x, 0)}, n_Q={sample_sizes_Q.get(x, 0)})"
        )

        # --- Filter continents with n_Metric < 10 ---
        valid_continents = {c for c, n in sample_sizes_reach.items() if n >= 9}
        diff_df = diff_df[diff_df['Continent'].isin(valid_continents)]

        
        # Determine hue order using reach-based counts
        hue_order = sorted(diff_df["Continent"].unique(), key=lambda x: sample_sizes_reach.get(x, 0), reverse=True)
        hue_order_n = [f"{c} (n_Metric={sample_sizes_reach.get(c, 0)}, n_Q={sample_sizes_Q.get(c, 0)})" for c in hue_order]

        # Custom pastel colors — adjust as needed
        custom_palette = {
            f'EU (n_Metric={sample_sizes_reach.get("EU", 0)}, n_Q={sample_sizes_Q.get("EU", 0)})': '#2ca02c',
            f'Southern NA (n_Metric={sample_sizes_reach.get("Southern NA", 0)}, n_Q={sample_sizes_Q.get("Southern NA", 0)})': '#ff7f0e',
            f'Northern NA (n_Metric={sample_sizes_reach.get("Northern NA", 0)}, n_Q={sample_sizes_Q.get("Northern NA", 0)})': '#1f77b4',
            f'SA (n_Metric={sample_sizes_reach.get("SA", 0)}, n_Q={sample_sizes_Q.get("SA", 0)})': 'palegoldenrod',
            f'OC (n_Metric={sample_sizes_reach.get("OC", 0)}, n_Q={sample_sizes_Q.get("OC", 0)})': 'purple',

        }

        filtered_palette = {k: v for k, v in custom_palette.items() if k in hue_order_n}

        if plot:
            plt.figure(figsize=(20, 9))
            ax = sns.boxplot(
                data=diff_df,
                x='Metric',
                y='Difference',
                hue='Continent (n)',
                hue_order=hue_order_n,
                palette=filtered_palette,
                showfliers=False
            )
            
            # Log10 difference lines at 2x, 3x, 5x and their inverses (0.5x, 0.33x, 0.2x)
            x_pos = len(list(diff_df['Metric'].unique())) - 0.4
            for val, label in zip([0, np.log10(2), np.log10(3), np.log10(5),
                                   -np.log10(2), -np.log10(3), -np.log10(5)],
                                  ['1x', '2x', '3x', '5x', '0.5x', '0.33x', '0.2x']):
                plt.axhline(val, linestyle='--', color='gray', linewidth=1.5, alpha=0.5)
                plt.text(x_pos-0.09, val, label, color='gray', fontsize=26, va='bottom')
            
            
            
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.title(f'Skill Difference by Continent: {run1} - {run2}', fontsize=34)
            plt.xlabel('Metric', fontsize=32)
            plt.ylabel('Difference', fontsize=32)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.grid(True, linestyle='--', alpha=0.5)

               # --- Improved significance star and sample size placement ---
            try:
                grouped = diff_df.groupby(['Metric', 'Continent'])
                metric_order = diff_df['Metric'].unique().tolist()
                continent_order = [c.split(" (n_")[0] for c in hue_order_n if " (n_" in c]
                spread = 0.3
                offset = -0.3

                for (metric, continent), group_df in grouped:
                    if continent not in continent_order:
                        continue

                    metric_idx = metric_order.index(metric)
                    continent_idx = continent_order.index(continent)
                    x_pos = metric_idx + offset + continent_idx * spread

                    # Retrieve significance p-value and sample size
                    pval = sig_labels.get((metric, continent), None)
                    n_reach = sample_sizes_reach.get(continent, 0)
                    n_q = sample_sizes_Q.get(continent, 0)

                    # Define star text
                    if pval is not None and pval < 0.05:
                        star_symbol = "*"
                        star_color = next((v for k, v in filtered_palette.items() if k.startswith(continent)), 'black')

                        # Find top of this continent's box distribution for metric
                        y_top = group_df['Difference'].quantile(0.75)
                        ax.text(
                            x_pos,
                            y_top + 0.2 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                            star_symbol,
                            ha='center', va='bottom',
                            fontsize=28,
                            color=star_color,
                            weight='bold'
                        )

                    # Add sample size label below boxes
                    # ax.text(
                    #     x_pos,
                    #     ax.get_ylim()[0] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    #     f"n={n_reach}/{n_q}",
                    #     ha='center',
                    #     va='top',
                    #     fontsize=10,
                    #     color='gray'
                    # )
            except Exception as e:
                print(f"Enhanced significance annotation failed: {e}")



            ax.legend(title='Continent', title_fontsize=24, fontsize=24, loc='best')
            plt.tight_layout()
            plt.savefig(f'//figs/gauge_metric_comparison_cont_{run1}_{run2}.png', dpi=350)
            plt.show()
            
            
#############
# Heatmap by reach_id
#################

def custom_coolwarm_gray_center():
    base = cm.get_cmap('coolwarm')(np.linspace(0, 1, 256))
    #base[125:131] = [0.7, 0.7, 0.7, 1]  # gray center
    return LinearSegmentedColormap.from_list("coolwarm_gray_center", base)


def plot_per_reach_metric_difference_heatmaps_switched_axes(dict_of_dfs, algo_name):
    metricList = ['r', 'NSE', 'nBIAS', '1-sigma']
    comparisons = [
        ('Fast', 'Continuous'),
        ('Science', 'Continuous'),
        ('Fast', 'Sampled'),
    ]

    vmin, vmax = -1, 1
    cmap = custom_coolwarm_gray_center()

    all_diffs = []
    all_reaches = set()

    # Collect all differences and all reach_ids
    for run1, run2 in comparisons:
        df1 = dict_of_dfs[run1]
        df2 = dict_of_dfs[run2]

        df1 = df1[df1['algo'] == algo_name][['reach_id'] + metricList].drop_duplicates('reach_id')
        df2 = df2[df2['algo'] == algo_name][['reach_id'] + metricList].drop_duplicates('reach_id')

        df1 = df1.rename(columns={m: f'{m}_1' for m in metricList})
        df2 = df2.rename(columns={m: f'{m}_2' for m in metricList})

        merged = pd.merge(df1, df2, on='reach_id', how='inner')

        diff_df = pd.DataFrame({'reach_id': merged['reach_id']})
        for metric in metricList:
            col1 = f'{metric}_1'
            col2 = f'{metric}_2'
            diff_df[metric] = (merged[col1].abs() - merged[col2].abs()) if metric == 'nBIAS' else merged[col1] - merged[col2]

        diff_df = diff_df.set_index('reach_id')
        all_diffs.append(diff_df)
        all_reaches.update(diff_df.index.tolist())

    # Identify valid reach_ids that have any real data across comparisons
    combined = pd.concat(all_diffs)
    reach_validity = combined.groupby(combined.index).apply(lambda df: df.notna().any().any())
    valid_reaches = reach_validity[reach_validity].index

    # Plot per comparison, with consistent X axis of valid_reaches only
    for diff_df, (run1, run2) in zip(all_diffs, comparisons):
        # Reindex to include all valid reaches, fill missing with NaNs
        diff_df = diff_df.reindex(valid_reaches)
        diff_df_t = diff_df.T

        clipped = diff_df_t.clip(lower=vmin, upper=vmax)

        plt.figure(figsize=(max(20, 0.8 * len(diff_df_t.columns)), max(16, 0.8 * len(diff_df_t.index))))
        ax = sns.heatmap(
            clipped,
            cmap=cmap,
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Difference'},
            vmin=vmin,
            vmax=vmax,
            square=False,
        )
        ax.collections[0].colorbar.set_label('Difference', fontsize=75)  # Change 14 to your desired font size
        # Enlarge colorbar tick labels
        ax.collections[0].colorbar.ax.tick_params(labelsize=75)

        
        # Get the colorbar and stretch it
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=70)
        cbar.set_label('Difference', fontsize=75)

        # Resize the colorbar using its axes (only works for default vertical bar)
        pos = cbar.ax.get_position()
        cbar.ax.set_position([pos.x0 - 0.4, pos.y0, pos.width + 0.4, pos.height])  # shift left & widen

        
        # Enlarge heatmap x- and y-tick labels
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=50, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=75, rotation=0)

        plt.title(f'Metric Differences per Reach: {run1} - {run2} ({algo_name})', fontsize=75)
        plt.xlabel('reach_id', fontsize=75)
        plt.xticks(fontsize=50)
        plt.yticks(rotation=45, fontsize=75)
        plt.tight_layout()
        plt.show()

        
###########
# Seasonal analysis
##################

def plot_orbitwise_log_ratio_vs_gauge_by_season(dfs_dict, consensus_label, gauge_label, plot=True):
    from scipy.stats import ttest_1samp, shapiro, wilcoxon
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'Siberia', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'Arctic', '9': 'GR'
    }
    
    full_season_order = ['4-7', '7-10', '10-1', '1-4']

    for orbit_name, df in dfs_dict.items():
        df = df.copy()
        df = df[df['algo'].isin([consensus_label, gauge_label])]
        if df.empty:
            continue

        df['time'] = pd.to_datetime(df['time'])
        pivoted = df.pivot_table(index=['reach_id', 'time'], columns='algo', values='Q')
        pivoted = pivoted.dropna(subset=[consensus_label, gauge_label])
        if pivoted.empty:
            continue

        # --- Compute reach-level medians before log difference ---
        pivoted = pivoted.reset_index()
        pivoted[consensus_label] = pd.to_numeric(pivoted[consensus_label], errors='coerce')
        pivoted[gauge_label] = pd.to_numeric(pivoted[gauge_label], errors='coerce')

        pivoted['continent'] = pivoted['reach_id'].astype(str).str[0]
        pivoted['continent_name'] = pivoted['continent'].map(continent_map)
        pivoted['season'] = pivoted['time'].apply(get_season_orbits)

        # Compute median discharge per reach per season
        reach_medians = (
            pivoted.groupby(['reach_id', 'continent_name', 'season'])
            [[consensus_label, gauge_label]]
            .median()
            .dropna()
            .reset_index()
        )

        reach_medians['log_ratio'] = np.log10(reach_medians[consensus_label]) - np.log10(reach_medians[gauge_label])
        reach_medians = reach_medians.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])

        # --- Filter seasons and set font sizes for Fast/Sampled orbits ---
        is_small_orbit = any(k in orbit_name for k in ['Sampled', 'Fast'])
        if is_small_orbit:
            season_order = [full_season_order[0]]  # Only first season
            reach_medians = reach_medians[reach_medians['season'] == full_season_order[0]]
            # Scaled down font sizes
            title_size = 20
            label_size = 18
            tick_size = 16
            legend_size = 14
            guide_size = 14
            star_size = 22
        else:
            season_order = full_season_order
            # Original font sizes
            title_size = 34
            label_size = 32
            tick_size = 28
            legend_size = 24
            guide_size = 24
            star_size = 38

        # --- Per-season & continent significance tests ---
        filtered_records, group_pvals, group_counts = [], {}, {}

        for (season, continent), group_df in reach_medians.groupby(['season', 'continent_name']):
            log_vals = group_df['log_ratio'].dropna()
            if len(log_vals) < 2:
                continue

            # Optional: apply modified_z_filter(log_vals) if available
            if len(log_vals) < 2:
                continue

            group_counts[(season, continent)] = len(log_vals)
            try:
                shapiro_p = shapiro(log_vals)[1]
                if shapiro_p > 0.05:
                    _, pval = ttest_1samp(log_vals, popmean=0)
                else:
                    _, pval = wilcoxon(log_vals, alternative='two-sided')
            except Exception:
                pval = np.nan

            group_pvals[(season, continent)] = pval

            for val in log_vals:
                filtered_records.append({
                    'season': season,
                    'continent_name': continent,
                    'log_ratio': val
                })

        if not filtered_records:
            continue

        # --- Filter for sufficient data per group ---
        valid_groups = {g for g, c in group_counts.items() if c >= 10}
        if not valid_groups:
            continue

        filtered_df = pd.DataFrame(filtered_records)
        filtered_df = filtered_df[
            filtered_df.apply(lambda r: (r['season'], r['continent_name']) in valid_groups, axis=1)
        ]
        filtered_df = filtered_df[
            ~filtered_df['continent_name'].isin(['SA', 'OC'])
        ]

        valid_continents = set(cont for _, cont in valid_groups)
        sample_sizes = {
            cont: reach_medians[reach_medians['continent_name'] == cont]['reach_id'].nunique()
            for cont in valid_continents
        }

        filtered_df["Continent (n)"] = filtered_df["continent_name"].apply(
            lambda x: f"{x} (n_gauge={sample_sizes.get(x, 0)})"
        )

        hue_order = sorted(
            filtered_df["Continent (n)"].unique(),
            key=lambda x: int(x.split('=')[1].rstrip(')')),
            reverse=False
        )

        custom_palette = {
            f"Arctic (n_gauge={sample_sizes.get('Arctic', 0)})": '#1f77b4',
            f"EU (n_gauge={sample_sizes.get('EU', 0)})": '#2ca02c',
            f"NA (n_gauge={sample_sizes.get('NA', 0)})": '#ff7f0e',
        }

        # --- Plot ---
        # Adjust figure width based on run name
        fig_width = 9 if is_small_orbit else 18
        plt.figure(figsize=(fig_width, 8))
        ax = sns.boxplot(
            data=filtered_df,
            x='season',
            y='log_ratio',
            hue='Continent (n)',
            palette=custom_palette,
            hue_order=hue_order,
            order=season_order
        )

        # Log10 ratio guide lines (1x, 2x, 3x, 5x)
        x_pos = len(filtered_df['season'].unique()) - 0.4
        for val, label in zip(
            [0, np.log10(2), np.log10(3), np.log10(5), -np.log10(2), -np.log10(3), -np.log10(5)],
            ['1x', '2x', '3x', '5x', '2x', '3x', '5x']
        ):
            plt.axhline(val, linestyle='--', color='black', linewidth=1.0, alpha=0.5)
            plt.text(x_pos - 0.1, val, label, color='black', fontsize=guide_size, va='bottom')

        plt.axhline(0.0, color='k', linestyle='--')
        plt.title(f"{orbit_name}: Consensus / Gauge", fontsize=title_size)
        plt.xlabel("Season", fontsize=label_size)
        if orbit_name in ['Continuous', 'Science']:
            plt.ylabel("Reach Q_cons / Q_gauge (log₁₀)", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.legend(fontsize=legend_size, loc='lower left') #bbox_to_anchor=(1.05, 1), 
        plt.ylim([-1.5, 1.5])
        plt.grid(False)
        plt.tight_layout()

        # --- Add significance stars ---
        season_list = sorted(filtered_df['season'].unique().tolist())
        continent_list = [label.split(' ')[0] for label in hue_order]
        grouped = filtered_df.groupby(['season', 'continent_name'])

        for (season, continent), group_df in grouped:
            if continent not in continent_list or season not in season_list:
                continue
            season_idx = season_list.index(season)
            continent_idx = continent_list.index(continent)
            spread = 0.3
            offset = -0.3
            x_pos = season_idx + offset + continent_idx * spread

            pval = group_pvals.get((season, continent), None)
            if pval is not None and pval < 0.05:
                label = f"{continent} (n_gauge={sample_sizes.get(continent, 0)})"
                star_color = custom_palette.get(label, "black")
                ax.text(x_pos, 1.2, '*', ha='center', va='bottom', fontsize=star_size, color=star_color, weight='bold')

        plt.savefig(f'//figs/gauge_cons_season_{orbit_name}_median.png',
                    bbox_inches='tight', dpi=350)
        plt.show()
        

################
# Regime Characterization Functions
##################

def summarize_overall_Q(dfs_q, algo):
    """
    Summarize Q statistics per reach_id including:
        - All Q90/Q10 values & dates
        - Counts, mean, SD
        - Original log10 stats and percentile summary
    """
    output = {}
    
    def process(df, run_label):
        df = df.copy()
        df = df[df['algo'] == algo]
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'Q', 'reach_id'])
        df = df[df['Q'] > 0]  # Only positive Q
        df['Q'] = pd.to_numeric(df['Q'], errors='coerce')
        df = df.dropna(subset=['Q'])
        df['logQ'] = np.log10(df['Q'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['logQ'])
        
        summaries = []
        for reach_id, group in df.groupby('reach_id'):
            Q_vals = group['Q'].values
            logQ_vals = group['logQ'].values
            time_vals = group['time'].values
            count = len(Q_vals)
            if count == 0:
                continue
            
            # Compute average interval between Q measurements (in days)
            if count > 1:
                intervals = np.diff(group['time']).astype('timedelta64[s]').astype(float) / (60*60*24)
                Q_interval = np.mean(intervals)
            else:
                Q_interval = np.nan
                
            # Original percentiles
            Q_p20 = np.percentile(Q_vals, 20)
            Q_p40 = np.percentile(Q_vals, 40)
            Q_p60 = np.percentile(Q_vals, 60)
            Q_p80 = np.percentile(Q_vals, 80)
            Q_p25 = np.percentile(Q_vals, 25)
            Q_p75 = np.percentile(Q_vals, 75)
            Q_median = np.median(Q_vals)
            Q_nIQR = (Q_p75 - Q_p25) / Q_median if Q_median != 0 else np.nan

            # Q10/Q90
            Q90_thresh = np.percentile(Q_vals, 90)
            Q10_thresh = np.percentile(Q_vals, 10)
            Q90_vals = Q_vals[Q_vals >= Q90_thresh]
            Q10_vals = Q_vals[Q_vals <= Q10_thresh]
            Q90_dates = time_vals[Q_vals >= Q90_thresh]
            Q10_dates = time_vals[Q_vals <= Q10_thresh]

            summary = {
                'reach_id': reach_id,
                'Q_log10_mean': np.mean(logQ_vals),
                'Q_log10_median': np.median(logQ_vals),
                'Q_log10_sd': np.std(logQ_vals),
                'Q_log10_max': np.max(logQ_vals),
                'Q_log10_min': np.min(logQ_vals),
                'Q_log10_IQR': (np.percentile(logQ_vals, 75) - np.percentile(logQ_vals, 25)) if count >= 10 else np.nan,
                'Q_log10_range': np.max(logQ_vals) - np.min(logQ_vals),                
                'Q_cv': (np.std(Q_vals) / np.mean(Q_vals)) if count >= 10 and np.mean(Q_vals) != 0 else np.nan,
                'Q_count': count,
                'Q_sum': np.sum(Q_vals),
                'Q_nIQR': Q_nIQR,
                'Q_p20': Q_p20,
                'Q_p40': Q_p40,
                'Q_p60': Q_p60,
                'Q_p80': Q_p80,
                'n_90': len(Q90_vals),
                'n_10': len(Q10_vals),
                'mean_90': np.mean(Q90_vals) if len(Q90_vals) > 0 else np.nan,
                'sd_90': np.std(Q90_vals) if len(Q90_vals) > 0 else np.nan,
                'mean_10': np.mean(Q10_vals) if len(Q10_vals) > 0 else np.nan,
                'sd_10': np.std(Q10_vals) if len(Q10_vals) > 0 else np.nan,
                'Q_interval': Q_interval
            }

            # Add each Q90/Q10 value & its date
            for i, (q, t) in enumerate(zip(Q90_vals, Q90_dates), start=1):
                summary[f'Q90_{i}'] = q
                summary[f'Q90_date_{i}'] = t
            for i, (q, t) in enumerate(zip(Q10_vals, Q10_dates), start=1):
                summary[f'Q10_{i}'] = q
                summary[f'Q10_date_{i}'] = t

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        summary_df['run'] = run_label
        summary_df['algo'] = algo
        return summary_df

    for run_label, df in dfs_q.items():
        output[run_label] = process(df, run_label)
        print('done', run_label)

    return output



def summarize_peaks(df, peak_prefixes=("Q90", "Q10"), date_tol=2):
    """
    Summarize dynamic peaks from Qxx_i and Qxx_date_i columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns like 'run', 'reach_id', Qxx_i, Qxx_date_i.
    peak_prefixes : tuple
        Prefixes of peak-related columns to process (default: ('Q90', 'Q10')).
    date_tol : int
        Tolerance in days to merge dates into a single peak.

    Returns
    -------
    pd.DataFrame
        One row per run–reach with peak summaries for each prefix.
    """

    results = []

    # Identify columns not related to peaks (metadata to preserve)
    peak_cols = [col for col in df.columns if any(col.startswith(p) for p in peak_prefixes)]
    meta_cols = [col for col in df.columns if col not in peak_cols]

    # Group by run and reach
    for (run, reach_id), group in df.groupby(["run", "reach_id"]):
        row = {"run": run, "reach_id": reach_id}

        # Preserve other metadata columns (first value in group)
        for col in meta_cols:
            row[col] = group[col].iloc[0]

        for prefix in peak_prefixes:
            # Collect all Q values and dates for this prefix
            values = []
            for col in group.columns:
                if col.startswith(prefix) and not "_date_" in col:
                    i = col.split("_")[-1]  # the suffix index
                    date_col = f"{prefix}_date_{i}"
                    if date_col in group.columns:
                        q = group[col].iloc[0]
                        d = pd.to_datetime(group[date_col].iloc[0])
                        if pd.notna(q) and pd.notna(d):
                            values.append((d, q))

            # Skip if no values
            if not values:
                continue

            # Sort by date
            values.sort(key=lambda x: x[0])

            # Merge into dynamic peaks
            peaks = []
            current = [values[0]]
            for d, q in values[1:]:
                if (d - current[-1][0]).days <= date_tol:
                    current.append((d, q))
                else:
                    peaks.append(current)
                    current = [(d, q)]
            peaks.append(current)

            # Compute stats for each peak
            for j, peak in enumerate(peaks, 1):
                dates, qs = zip(*peak)

                # find max/min
                max_q = max(qs)
                min_q = min(qs)
                max_date = dates[qs.index(max_q)]
                min_date = dates[qs.index(min_q)]

                row[f"{prefix}_peak{j}_start"] = min(dates)
                row[f"{prefix}_peak{j}_end"] = max(dates)
                row[f"{prefix}_peak{j}_max"] = max_q
                row[f"{prefix}_peak{j}_max_date"] = max_date
                row[f"{prefix}_peak{j}_min"] = min_q
                row[f"{prefix}_peak{j}_min_date"] = min_date
                row[f"{prefix}_peak{j}_mean"] = sum(qs) / len(qs)
                row[f"{prefix}_peak{j}_duration"] = (max(dates) - min(dates)).days + 1

        results.append(row)

    return pd.DataFrame(results)


def summarize_overall_Q_by_season(dfs_q, algo):
    """
    Summarize Q statistics per reach_id and season.
    Adds seasonal suffix to stats (e.g., Q_log10_mean_1-4, Q_log10_mean_7-10).
    """
    output = {}

    def process(df, run_label):
        df = df.copy()
        df = df[df['algo'] == algo]
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'Q', 'reach_id'])
        df = df[df['Q'] > 0]  # Only positive Q
        df['Q'] = pd.to_numeric(df['Q'], errors='coerce')
        df = df.dropna(subset=['Q'])
        df['logQ'] = np.log10(df['Q'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['logQ'])

        # Assign season
        df['season'] = df['time'].apply(get_season_orbits)

        summaries = []
        for (reach_id, season), group in df.groupby(['reach_id', 'season']):
            Q_vals = group['Q'].values
            logQ_vals = group['logQ'].values
            count = len(Q_vals)
            if count == 0:
                continue

            # Compute average interval between Q measurements (days)
            if count > 1:
                intervals = np.diff(group['time']).astype('timedelta64[s]').astype(float) / (60 * 60 * 24)
                Q_interval = np.mean(intervals)
            else:
                Q_interval = np.nan

            # Original percentiles
            Q_p20 = np.percentile(Q_vals, 20)
            Q_p40 = np.percentile(Q_vals, 40)
            Q_p60 = np.percentile(Q_vals, 60)
            Q_p80 = np.percentile(Q_vals, 80)
            Q_p25 = np.percentile(Q_vals, 25)
            Q_p75 = np.percentile(Q_vals, 75)
            Q_median = np.median(Q_vals)
            Q_nIQR = (Q_p75 - Q_p25) / Q_median if Q_median != 0 else np.nan

            # Build seasonal summary (exclude Q90/Q10 details here)
            summary = {
                'reach_id': reach_id,
                f'Q_log10_mean_{season}': np.mean(logQ_vals),
                f'Q_log10_median_{season}': np.median(logQ_vals),
                f'Q_log10_sd_{season}': np.std(logQ_vals),
                f'Q_log10_max_{season}': np.max(logQ_vals),
                f'Q_log10_min_{season}': np.min(logQ_vals),
                f'Q_log10_IQR_{season}': (np.percentile(logQ_vals, 75) - np.percentile(logQ_vals, 25)) if count >= 10 else np.nan,
                f'Q_log10_range_{season}': np.max(logQ_vals) - np.min(logQ_vals),
                f'Q_cv_{season}': (np.std(Q_vals) / np.mean(Q_vals)) if count >= 10 and np.mean(Q_vals) != 0 else np.nan,
                f'Q_count_{season}': count,
                f'Q_sum_{season}': np.sum(Q_vals),
                f'Q_nIQR_{season}': Q_nIQR,
                f'Q_p20_{season}': Q_p20,
                f'Q_p40_{season}': Q_p40,
                f'Q_p60_{season}': Q_p60,
                f'Q_p80_{season}': Q_p80,
                f'Q_interval_{season}': Q_interval,
            }
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        if not summary_df.empty:
            summary_df['run'] = run_label
            summary_df['algo'] = algo
        return summary_df

    for run_label, df in dfs_q.items():
        output[run_label] = process(df, run_label)
        print('done', run_label)

    return output


#####
# FDC
#####

def plot_reach_consensus_cdfs(df_dict, variable, algo, color_dict):
    """
    Plot consensus CDFs for each orbit/run on one plot per reach.
    
    Parameters:
        df_dict: dict of DataFrames, keyed by orbit/run label
        variable: str, variable to analyze (e.g., 'Q')
        algo: str, algorithm name to filter (e.g., 'consensus')
        color_dict: dict mapping run labels -> colors
    """
    
    # Store summary metrics
    summary_rows = []
    required_runs = {"Fast", "Science", "Continuous"}
    # Get all unique reach_ids across all dataframes
    all_reaches = sorted(set().union(*[df['reach_id'].unique() for df in df_dict.values()]))
    specific_reaches = [78322500121, 72552000291, 73120000131, 23214400041]
    for reach_id in specific_reaches:
        plt.figure(figsize=(12, 8))
        data_dict = {}
        
        # --- Loop over runs/orbits ---
        for run_label, df in df_dict.items():
            # if run_label in ['Continuous-FSO', 'Continuous-SO']:
            #     continue
                
            df_reach = df[(df['reach_id'] == reach_id) & (df['algo'] == algo)]
            if df_reach.empty:
                continue
            
            data = df_reach[variable].dropna().to_numpy()
            if len(data) < 30:
                continue
            
            data = np.sort(data)
            cdf = np.arange(1, len(data) + 1) / len(data)
            
            color = color_dict.get(run_label, "grey")
            plt.plot(data, cdf, label=f"{run_label} (n={len(data)})", color=color, linewidth=4)
            data_dict[run_label] = data
            
 # Overlay gauge ONLY for Continuous
            if run_label == "Continuous":
                df_gauge = df[(df['reach_id'] == reach_id) & (df['algo'] == "gauge")]
                if not df_gauge.empty:
                    data = df_gauge[variable].dropna().to_numpy()
                    if len(data) > 0:
                        data = np.sort(data)
                        cdf = np.arange(1, len(data) + 1) / len(data)
                        plt.plot(data, cdf, label=f"{run_label} (gauge, n={len(data)})",
                                 color="black", linewidth=3, linestyle="--")
                        # Store under a combined key
                        data_dict[f"{run_label}_gauge"] = data            
            
            
            
        # --- Require Fast, Science, and Continuous ---
        if not required_runs.issubset(data_dict.keys()):
            plt.close()
            continue      
        # --- Compute pairwise distances ---
        run_labels = list(data_dict.keys())
        dist_text = ""
        for i in range(len(run_labels)):
            for j in range(i + 1, len(run_labels)):
                a1, a2 = run_labels[i], run_labels[j]
                d1, d2 = data_dict[a1], data_dict[a2]

                ks_stat, p_value = ks_2samp(d1, d2)  # <-- get p-value too
                combined = np.sort(np.unique(np.concatenate([d1, d2])))
                cdf1 = np.searchsorted(d1, combined, side='right') / len(d1)
                cdf2 = np.searchsorted(d2, combined, side='right') / len(d2)
                l1_dist = np.sum(np.abs(cdf1 - cdf2) * np.diff(np.concatenate([[combined[0]-1], combined])))
                emd = wasserstein_distance(d1, d2)

                dist_text += f"{a1}-{a2} KS/p_value={ks_stat:.2f}/{p_value:.2f}, L1={l1_dist:.2f}\n" #, EMD={emd:.2f}\n"

                summary_rows.append({
                    'reach_id': reach_id,
                    'pair': f"{a1}-{a2}",
                    'KS': ks_stat,
                    'p_value': p_value,
                    'L1': l1_dist,
                    'EMD': emd
                })
        
        # --- Final formatting ---
        plt.title(f"Consensus FDC — Reach {reach_id}", fontsize=32)
        plt.xlabel(f"Discharge (m3/s)", fontsize=28)
        plt.ylabel("Proportion", fontsize=28)
        plt.grid(True)
        plt.legend(loc="lower right", fontsize=26)
        
        # Put distances in a text box
        if dist_text:
            plt.gcf().text(
                0.5, -0.05, dist_text,
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black')
            )
        
        plt.savefig(f'/fdc_{reach_id}.png', dpi=350, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def plot_reach_cdfs_horizontal_with_ks_emd(df_dict, variable, algos):
    """
    Plot horizontal 4-panel CDFs for each reach and compute KS, L1, and EMD
    between each algorithm pair. Also return a summary table of distances.
    
    Parameters:
        df_dict: dict of DataFrames, keyed by orbit/run label
        variable: str, variable to analyze (e.g., 'Q')
        algos: list of algorithm names to compare
        
    Returns:
        summary_df: pd.DataFrame with reach_id as rows and distance metrics as columns
    """
    
    # Store summary metrics
    summary_rows = []

    # Get all unique reach_ids across all dataframes
    all_reaches = set()
    for df in df_dict.values():
        all_reaches.update(df['reach_id'].unique())
    all_reaches = sorted(all_reaches)
    
    for reach_id in all_reaches:
        fig, axes = plt.subplots(1, 4, figsize=(25, 8), sharey=True)
        axes = axes.flatten()
        
        for i, (run_label, df) in enumerate(df_dict.items()):
            if i >= 4:
                break  # Only plot up to 4 runs per reach
            ax = axes[i]
            df_reach = df[df['reach_id'] == reach_id]
            
            data_dict = {}
            for algo in algos:
                if algo in df_reach['algo'].values:
                    data = df_reach[df_reach['algo'] == algo][variable].dropna().to_numpy()
                    n_points = len(data)
                    if n_points == 0:
                        continue
                    data_dict[algo] = np.sort(data)
                    cdf = np.arange(1, n_points + 1) / n_points
                    ax.plot(data_dict[algo], cdf, label=f"{algo} (n={n_points})")
            
            ax.set_title(f"{run_label} — Reach {reach_id}", fontsize=14)
            ax.set_xlabel(variable)
            ax.grid(True)
            if i == 0:
                ax.set_ylabel('CDF', fontsize=14)
            
            # Show legend on last panel only
            if i == 3:
                ax.legend(loc='lower right', fontsize=12)
            
            # Compute KS, L1, and EMD for each pair
            pairs = [(algos[0], algos[1]), (algos[0], algos[2]), (algos[1], algos[2])]
            dist_text = ""
            for a1, a2 in pairs:
                if a1 in data_dict and a2 in data_dict:
                    # KS statistic
                    ks_stat = ks_2samp(data_dict[a1], data_dict[a2]).statistic
                    # L1 distance (approximate integral of |CDF1 - CDF2|)
                    combined = np.sort(np.unique(np.concatenate([data_dict[a1], data_dict[a2]])))
                    cdf1 = np.searchsorted(data_dict[a1], combined, side='right') / len(data_dict[a1])
                    cdf2 = np.searchsorted(data_dict[a2], combined, side='right') / len(data_dict[a2])
                    l1_dist = np.sum(np.abs(cdf1 - cdf2) * np.diff(np.concatenate([[combined[0]-1], combined])))
                    # Earth Mover's Distance (Wasserstein)
                    emd = wasserstein_distance(data_dict[a1], data_dict[a2])
                    dist_text += f"{a1}-{a2} KS={ks_stat:.2f} L1={l1_dist:.2f} EMD={emd:.2f}\n"

                    # Add to summary table
                    summary_rows.append({
                        'reach_id': reach_id,
                        'orbit': run_label,
                        'pair': f"{a1}-{a2}",
                        'KS': ks_stat,
                        'L1': l1_dist,
                        'EMD': emd
                    })
            
            # Place distance text in a box below each panel
#             ax.text(
#                 0.5, -0.3, dist_text,
#                 transform=ax.transAxes,
#                 fontsize=12,
#                 ha='center',
#                 va='top',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black')
#             )
        
#         plt.tight_layout()
#         plt.show()
    
    # Convert summary_rows to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


#######
# Revisit times
##############

    
def get_revisit_times(dfs_q, color_dict_timeseries):
    seasons = ["4-7", "7-10", "10-1", "1-4"]

    for season in seasons:
        # Prepare a list to collect results
        scatter_data = []
        cdf_data = {}
        reach_counts = {}  # store n per df_name

        for df_name, df in dfs_q.items():
            df = df.copy()
            df['season'] = pd.to_datetime(df['time']).apply(get_season_orbits)
            if df_name in ['Continuous-FSO', 'Continuous-SO']:
                continue

            if df_name in ['Continuous', 'Science', 'Fast', 'Sampled']:
                df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

                # keep only this season
                df = df[df['season'] == season]
                if season == '4-7':
                    df = df[df['time'] < '2025-03-30']
                    
            # Filter consensus only
            df_cons = df[df['algo'] == 'consensus'].copy()
            df_cons = df_cons.sort_values(['reach_id', 'time'])
            print(season, df_name, df_cons.reach_id.nunique())

            revisit_periods_reach = []

            # Compute revisit period per reach
            for reach, group in df_cons.groupby('reach_id'):
                times = pd.to_datetime(group['time']).sort_values()
                
                if df_name in ['Continuous', 'Science']: 
                    if len(times) > 10:
                        # add year and season columns
                        group = group.copy()
                        group['year'] = group['time'].dt.year
                        group['season'] = group['time'].apply(get_season_orbits)

                        # compute revisit periods within (year, season)
                        deltas = []
                        for (yr, ssn), sub in group.groupby(['year', 'season']):
                            if len(sub) > 1:
                                delta_days = sub['time'].sort_values().diff().dt.total_seconds().dropna() / 86400.0
                                deltas.extend(delta_days.tolist())

                        if deltas:
                            revisit_period = np.mean(deltas)
                            mean_flow = group['Q'].mean()
                            scatter_data.append({
                                'df_name': df_name,
                                'reach_id': reach,
                                'revisit_period': revisit_period,
                                'mean_flow': mean_flow
                            })
                            revisit_periods_reach.append(revisit_period)
                
                if df_name in ['Sampled', 'Fast']: 
                    if len(times) > 10:
                        # add year and season columns
                        group = group.copy()
                        group['year'] = group['time'].dt.year
                        group['season'] = group['time'].apply(get_season_orbits)

                        # compute revisit periods within (year, season)
                        deltas = []
                        for (yr, ssn), sub in group.groupby(['year', 'season']):
                            if len(sub) > 1:
                                delta_days = sub['time'].sort_values().diff().dt.total_seconds().dropna() / 86400.0
                                deltas.extend(delta_days.tolist())

                        if deltas:
                            revisit_period = np.mean(deltas)
                            mean_flow = group['Q'].mean()
                            scatter_data.append({
                                'df_name': df_name,
                                'reach_id': reach,
                                'revisit_period': revisit_period,
                                'mean_flow': mean_flow
                            })
                            revisit_periods_reach.append(revisit_period)

            if revisit_periods_reach:
                cdf_data[df_name] = np.sort(revisit_periods_reach)
                reach_counts[df_name] = len(revisit_periods_reach)

        # Convert to DataFrame for scatter plot
        scatter_df = pd.DataFrame(scatter_data)
        print(season, scatter_df.groupby(['df_name']).revisit_period.describe())

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 15))

        # Scatter plot
        for df_name, group in scatter_df.groupby('df_name'):
            n_reaches = reach_counts.get(df_name, 0)
            axes[0].scatter(
                group['revisit_period'], 
                group['mean_flow'], 
                s=10,
                label=f"{df_name} (n={n_reaches})", 
                color=color_dict_timeseries.get(df_name, 'black'),
                alpha=0.5
            )

        axes[0].set_yscale('log')
        axes[0].set_xlabel('Revisit Period (days)')
        axes[0].set_ylabel('Mean Flow (Q)')
        axes[0].set_title(f'Revisit Period vs Mean Flow per Reach — {season}')
        axes[0].legend()
        axes[0].grid(True)

        # CDF plot
        for df_name, sorted_rp in cdf_data.items():
            n_reaches = reach_counts.get(df_name, 0)
            cdf = np.arange(1, len(sorted_rp)+1) / len(sorted_rp)
            axes[1].plot(
                sorted_rp, 
                cdf, 
                label=f"{df_name} (n={n_reaches})", 
                color=color_dict_timeseries.get(df_name, 'black')
            )
        axes[1].axhline(y=0.50, color='black', linestyle='dashed')
        axes[1].set_xlabel('Revisit Period (days)')
        axes[1].set_ylabel('CDF')
        axes[1].set_title(f'CDF of Revisit Period per Orbit — {season}')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.xlim([0,50])
        plt.tight_layout()
        plt.show()

def get_revisit_times_overall(dfs_q, color_dict_timeseries):
    # Prepare a list to collect results
    scatter_data = []
    cdf_data = {}
    reach_counts = {}  # store n per df_name

    for df_name, df in dfs_q.items():
        df = df.copy()
        if df_name in ['Continuous-FSO', 'Continuous-SO']:
            continue

        if df_name in ['Continuous', 'Science', 'Fast', 'Sampled']:
            df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

        # Filter consensus only
        df_cons = df[df['algo'] == 'consensus'].copy()
        df_cons = df_cons.sort_values(['reach_id', 'time'])
        print(df_name, df_cons.reach_id.nunique())

        revisit_periods_reach = []

        # Compute revisit period per reach
        for reach, group in df_cons.groupby('reach_id'):
            times = pd.to_datetime(group['time']).sort_values()

            if df_name in ['Continuous', 'Science']:
                if len(times) > 10:
                    delta_days = times.diff().dt.total_seconds().dropna() / 86400.0
                    revisit_period = delta_days.mean()
                    mean_flow = group['Q'].mean()
                    scatter_data.append({
                        'df_name': df_name,
                        'reach_id': reach,
                        'revisit_period': revisit_period,
                        'mean_flow': mean_flow
                    })
                    revisit_periods_reach.append(revisit_period)

            elif df_name in ['Fast', 'Sampled']:
                if len(times) > 10:
                    delta_days = times.diff().dt.total_seconds().dropna() / 86400.0
                    revisit_period = delta_days.mean()
                    mean_flow = group['Q'].mean()
                    scatter_data.append({
                        'df_name': df_name,
                        'reach_id': reach,
                        'revisit_period': revisit_period,
                        'mean_flow': mean_flow
                    })
                    revisit_periods_reach.append(revisit_period)

        if revisit_periods_reach:
            cdf_data[df_name] = np.sort(revisit_periods_reach)
            reach_counts[df_name] = len(revisit_periods_reach)

    # Convert to DataFrame for scatter plot
    scatter_df = pd.DataFrame(scatter_data)
    print(scatter_df.groupby(['df_name']).revisit_period.describe())

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    # Scatter plot
    for df_name, group in scatter_df.groupby('df_name'):
        n_reaches = reach_counts.get(df_name, 0)
        axes[0].scatter(
            group['revisit_period'], 
            group['mean_flow'], 
            s=10,
            label=f"{df_name} (n={n_reaches})", 
            color=color_dict_timeseries.get(df_name, 'black'),
            alpha=0.5
        )

    axes[0].set_yscale('log')
    axes[0].set_xlabel('Revisit Period (days)')
    axes[0].set_ylabel('Mean Flow (Q)')
    axes[0].set_title('Revisit Period vs Mean Flow per Reach — Overall')
    axes[0].legend()
    axes[0].grid(True)

    # CDF plot
    for df_name, sorted_rp in cdf_data.items():
        n_reaches = reach_counts.get(df_name, 0)
        cdf = np.arange(1, len(sorted_rp)+1) / len(sorted_rp)
        axes[1].plot(
            sorted_rp, 
            cdf, 
            label=f"{df_name} (n={n_reaches})", 
            color=color_dict_timeseries.get(df_name, 'black')
        )
    axes[1].axhline(y=0.50, color='black', linestyle='dashed')
    axes[1].set_xlabel('Revisit Period (days)')
    axes[1].set_ylabel('CDF')
    axes[1].set_title('CDF of Revisit Period per Orbit — Overall')
    axes[1].legend()
    axes[1].grid(True)
    plt.xlim([0,50])
    plt.tight_layout()
    plt.show()



def find_overlaps(peaks, reference_algo="gauge", tol=pd.Timedelta(days=1)):
    """Find overlapping peaks between reference algo and others, compute differences.
       Also record gauge peaks with no overlap (misses), and assign season.
       Keeps start, end, min_date, and max_date of both reference and overlapping peaks.
    """
    results = []
    for run in peaks["run"].unique():
        run_peaks = peaks[peaks["run"] == run]
        gauge_peaks = run_peaks[run_peaks["algo"] == reference_algo]
        others = run_peaks[run_peaks["algo"] != reference_algo]

        for _, g in gauge_peaks.iterrows():
            same_reach = others[others["reach_id"] == g["reach_id"]]

            # Assign season based on gauge peak start
            season = get_season_orbits(g["start"])

            overlaps = same_reach[
                (same_reach["start"] <= g["end"] + tol) &
                (same_reach["end"] >= g["start"] - tol)
            ]

            if overlaps.empty:
                # record a miss for each algo
                for algo in same_reach["algo"].unique():
                    results.append({
                        "run": run,
                        "reach_id": g["reach_id"],
                        "algo": algo,
                        "matched": False,
                        "duration_diff": None,
                        "g_duration": None,
                        "o_duration": None,
                        "g_mean": None,
                        "o_mean": None,
                        "max_date_diff_days": None,
                        "min_date_diff_days": None,
                        "g_start": g["start"],
                        "g_end": g["end"],
                        "o_start": None,
                        "o_end": None,
                        "g_min_date": g.get("min_date", None),
                        "g_max_date": g.get("max_date", None),
                        "o_min_date": None,
                        "o_max_date": None,
                        "season": season
                    })
            else:
                for _, o in overlaps.iterrows():
                    results.append({
                        "run": run,
                        "reach_id": g["reach_id"],
                        "algo": o["algo"],
                        "matched": True,
                        "duration_diff": (
                            o["duration"] - g["duration"]
                            if pd.notna(g["duration"]) and pd.notna(o["duration"])
                            else None
                        ),
                        "g_duration": g["duration"],
                        "o_duration": o["duration"],
                        "g_mean": g['mean'],
                        "o_mean": o['mean'],
                        "max_date_diff_days": (
                            (o["max_date"] - g["max_date"]).days
                            if pd.notna(g["max_date"]) and pd.notna(o["max_date"])
                            else None
                        ),
                        "min_date_diff_days": (
                            (o["min_date"] - g["min_date"]).days
                            if pd.notna(g["min_date"]) and pd.notna(o["min_date"])
                            else None
                        ),
                        "g_start": g["start"],
                        "g_end": g["end"],
                        "o_start": o["start"],
                        "o_end": o["end"],
                        "g_min_date": g.get("min_date", None),
                        "g_max_date": g.get("max_date", None),
                        "o_min_date": o.get("min_date", None),
                        "o_max_date": o.get("max_date", None),
                        "season": season
                    })
    return pd.DataFrame(results)



############
#Percentile Analysis
##############
def plot_q90_summary_by_run(overall_summary, peaks_long, overlaps, color_dict_timeseries, peak_type):
    # --- Filter gauge and consensus ---
    df = overall_summary[overall_summary["algo"].isin(["gauge", "consensus"])].copy()

    # --- Number of Q90 peaks ---
    q90_peak_cols = [c for c in df.columns if c.startswith(f"Q{peak_type}_peak") and c.endswith("_start")]
    df[f"n_Q{peak_type}_peaks"] = df[q90_peak_cols].notna().sum(axis=1)

    # --- Durations ---
    duration_cols = [c for c in df.columns if c.startswith(f"Q{peak_type}_peak") and c.endswith("_duration")]
    df_durations = df.melt(
        id_vars=["run", "algo"], 
        value_vars=duration_cols,
        var_name="peak", 
        value_name="duration"
    ).dropna(subset=["duration"])
    df_durations = df_durations[df_durations['duration'] < 14.0]
    
    # --- Reshape peaks for duration and dates --- 
    peaks_long = reshape_peaks(df, peak_type=peak_type) 
   # peaks_long = peaks_long[peaks_long['duration'] < 14.0]
    # print(peaks_long)
    overlaps = find_overlaps(peaks_long, reference_algo="gauge")
    overlaps = overlaps.dropna(subset="max_date_diff_days")
    #print(overlaps.head(20))
    
    df_o_max = overlaps[["run", "algo", "max_date_diff_days"]].rename(columns={"max_date_diff_days":"value"})
    df_o_max["metric"] = "Q_max Date Difference (days)"
    print("Q_max Date Difference (days)", df_o_max.groupby(['run'])['value'].describe())
          
    # --- Mean flows ---
    # mean_cols = [c for c in df.columns if c.startswith(f"Q{peak_type}_peak") and c.endswith("_mean")]
    # df_means = df.melt(
    #     id_vars=["run", "algo"],
    #     value_vars=mean_cols,
    #     var_name="peak",
    #     value_name="value"
    # ).dropna(subset=["value"])
    # print(df_means)
    # print(df_means.groupby(['run'])['value'].describe())

    # --- g_duration rows ---
    df_g = overlaps.copy()
    df_g["algo"] = "gauge"
    df_g["duration"] = df_g["g_duration"]
    df_g = df_g.drop(columns=["g_duration", "o_duration"])

    # --- o_mean rows ---
    df_o = overlaps.copy()
    df_o["algo"] = "consensus"
    df_o["duration"] = df_o["o_duration"]
    df_o = df_o.drop(columns=["g_duration", "o_duration"])

    # --- concatenate ---
    df_long = pd.concat([df_g, df_o], axis=0, ignore_index=True)
    df_durations = df_long[["run", "algo", "duration"]].rename(columns={"duration":"value"})
    df_durations["metric"] = f"T{peak_type} Duration (days)"
    print(f"T{peak_type} Duration (days)", df_durations[df_durations['algo'] == 'consensus'].groupby(['run'])['value'].describe())
    print(f"T{peak_type} Duration (days)", df_durations[df_durations['algo'] == 'gauge'].groupby(['run'])['value'].describe())

    # --- g_mean rows ---
    df_g = overlaps.copy()
    df_g["algo"] = "gauge"
    df_g["mean"] = df_g["g_mean"]
    df_g = df_g.drop(columns=["g_mean", "o_mean"])

    # --- o_mean rows ---
    df_o = overlaps.copy()
    df_o["algo"] = "consensus"
    df_o["mean"] = df_o["o_mean"]
    df_o = df_o.drop(columns=["g_mean", "o_mean"])

    # --- concatenate ---
    df_long = pd.concat([df_g, df_o], axis=0, ignore_index=True)
    df_means = df_long[["run", "algo", "mean"]].rename(columns={"mean":"value"})
    df_means["metric"] = f"Q{peak_type} Mean Flow (m3/s)"
    print(f"Q{peak_type} Mean Flow (m3/s)", df_means[df_means['algo'] == 'consensus'].groupby(['run'])['value'].describe())
    print(f"Q{peak_type} Mean Flow (m3/s)", df_means[df_means['algo'] == 'gauge'].groupby(['run'])['value'].describe())
    
    # df_o_means = overlaps[["run", "algo", "g_mean","o_mean"]].rename(columns={"mean":"value"})
    # df_o_means["metric"] = "Q90 Mean Flow (m3/s)"
    # print(df_o_means, df_o_means.groupby(['run'])['value'].describe())
    
    # --- Combine linear metrics ---
    df_n = df[["run", "algo", f"n_Q{peak_type}_peaks"]].rename(columns={f"n_Q{peak_type}_peaks":"value"})
    df_n["metric"] = f"Q{peak_type} Instances/Reach"
    print(f"Q{peak_type} Instances/Reach", df_n[df_n['algo'] == 'consensus'].groupby(['run'])['value'].describe())
    print(f"Q{peak_type} Instances/Reach", df_n[df_n['algo'] == 'gauge'].groupby(['run'])['value'].describe())

    # df_d = df_durations[["run", "algo", "duration"]].rename(columns={"duration":"value"})
    # df_d["metric"] = "T90 Duration (days)"

    # Linear metrics (first two groups) go on left y-axis
    df_linear = pd.concat([df_n, df_durations, df_o_max], axis=0, ignore_index=True).reset_index(drop=True)
    #print(df_linear)
    
    # --- Run order ---
    run_order = ['Continuous', 'Fast', 'Science', 'Sampled']
    df_linear['run'] = pd.Categorical(df_linear['run'], categories=run_order, ordered=True)
    overlaps['run'] = pd.Categorical(overlaps['run'], categories=run_order, ordered=True)
    df_means['run'] = pd.Categorical(df_means['run'], categories=run_order, ordered=True)

    # --- Figure ---
    fig, ax1 = plt.subplots(figsize=(40,18))
    ax2 = ax1.twinx()  # log axis for Q90 Mean Flow

    width = 0.5
    spacing = 0.5  # gap between gauge and consensus

    # --- Plot first two linear metrics ---
    linear_metrics = [f"Q{peak_type} Instances/Reach", f"T{peak_type} Duration (days)", "Q_max Date Difference (days)"]
    for m_idx, metric in enumerate(linear_metrics):
        x_base = m_idx * (len(run_order)*1.2 + 1)
        for r_idx, run in enumerate(run_order):
            # Positions
            x_gauge = x_base + r_idx*1.2
            x_cons = x_base + r_idx*1.2 + spacing

            # Values
            gauge_vals = df_linear[(df_linear['metric'] == metric) &
                                   (df_linear['run'] == run) &
                                   (df_linear['algo'] == 'gauge')]['value']
            cons_vals = df_linear[(df_linear['metric'] == metric) &
                                  (df_linear['run'] == run) &
                                  (df_linear['algo'] == 'consensus')]['value']

            # Plot
            ax1.boxplot(gauge_vals, positions=[x_gauge], widths=width, patch_artist=True,
                        boxprops=dict(facecolor='grey'), medianprops=dict(color='black'))
            ax1.boxplot(cons_vals, positions=[x_cons], widths=width, patch_artist=True,
                        boxprops=dict(facecolor=color_dict_timeseries[run]), medianprops=dict(color='black'))



    # --- Plot Q90 Mean Flow (last group, log axis) ---
    metric = f"Q{peak_type} Mean Flow (m3/s)"
    x_base = (len(linear_metrics)) * (len(run_order)*1.2 + 1)  # after third group
    for r_idx, run in enumerate(run_order):
        x_gauge = x_base + r_idx*1.2
        x_cons = x_base + r_idx*1.2 + spacing

        gauge_vals = df_means[(df_means['run'] == run) &
                               (df_means['algo'] == 'gauge')]['value']
        cons_vals = df_means[(df_means['run'] == run) &
                              (df_means['algo'] == 'consensus')]['value']

        ax2.boxplot(gauge_vals, positions=[x_gauge], widths=width, patch_artist=True,
                    boxprops=dict(facecolor='grey'), medianprops=dict(color='black'))
        ax2.boxplot(cons_vals, positions=[x_cons], widths=width, patch_artist=True,
                    boxprops=dict(facecolor=color_dict_timeseries[run]), medianprops=dict(color='black'))
    ax2.set_yscale('log')

    # --- X-axis labels ---
    all_metrics = linear_metrics + [f"Q{peak_type} Mean Flow (m3/s)"]
    xticks = []
    xticklabels = []    
    group_positions = []  # store midpoints for metric labels
    for m_idx, metric in enumerate(all_metrics):
        x_base = m_idx * (len(run_order)*1.25 + 1)
        run_positions = [x_base + r_idx*1.2 for r_idx in range(len(run_order))]
        group_positions.append(np.mean(run_positions))  # center of the group

        for r_idx, run in enumerate(run_order):
            x_tick = x_base + r_idx*1.2
            xticks.append(x_tick)
            xticklabels.append(f"{run}")

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, rotation=45, fontsize = 16, ha='right')
    
    # --- Add sample size (n=...) below each run label ---
    for m_idx, metric in enumerate(all_metrics):
        x_base = m_idx * (len(run_order)*1.25 + 1)
        for r_idx, run in enumerate(run_order):
            # get subset
            if metric == f"Q{peak_type} Mean Flow (m3/s)":
                subset = df_means[(df_means['run'] == run)]
                values = subset["value"]
                reach_ids = subset.get("reach_id", None)
            else:
                subset = df_linear[(df_linear['metric'] == metric) &
                                   (df_linear['run'] == run)]
                values = subset["value"]
                reach_ids = subset.get("reach_id", None)

            # compute n as unique reach_id count if present, else count values
            if reach_ids is not None and not reach_ids.isna().all():
                n_val = reach_ids.nunique()
            else:
                n_val = values.count()

            # position text just below x tick
            x_pos = x_base + r_idx*1.2#01 + spacing/2.4
            y_min = ax1.get_ylim()[0]
            # ax1.text(
            #     x_pos, y_min+48, f"n={n_val}",
            #     ha='center', va='top', fontsize=30, color="black", rotation=0
            # )
            
    # --- Add secondary x-axis for metric group labels ---
    ax3 = ax1.twiny()  
    ax3.set_xlim(ax1.get_xlim())  # align with primary axis
    ax3.set_xticks(group_positions)
    ax3.set_xticklabels(all_metrics, fontsize=85, fontweight='bold')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.xaxis.set_label_position('bottom')
    ax3.spines['bottom'].set_position(('outward', 60))  # shift below run labels
    ax3.spines['bottom'].set_visible(False)  # hide spine line
    
    ax1.set_ylabel("Value", fontsize=46)
    ax1.tick_params(axis='both', labelsize=34)
    ax3.tick_params(axis='x', labelsize=46, pad=95)

    ax1.set_ylim(-10, 30)
    ax1.set_yticks(np.arange(-10, 31, 5))
    
    ax2.set_ylabel(f"Q{peak_type} Mean Flow (Log)", fontsize=46)
    ax2.tick_params(axis='y', labelsize=40)

    #ax1.set_xlabel("Metric / Run")
    #ax1.set_title(f"Q{peak_type} Capture Summary")
    ax1.grid(True, axis='y', linewidth = 1, linestyle='dotted')
    ax2.grid(True, axis='y', linewidth = 1)
    ax1.grid(False, axis='x')
    ax2.grid(False, axis='x')
    ax3.grid(False, axis='x')

    import matplotlib.patches as mpatches

    # --- Build legend handles from color_dict_timeseries ---
    legend_elements = [
        mpatches.Patch(facecolor='grey', label='Gauge')
    ]

    # Add run colors
    for run in run_order:
        if run in color_dict_timeseries:
            legend_elements.append(
                mpatches.Patch(facecolor=color_dict_timeseries[run], label=f'{run}')
            )

    # Place legend
    ax1.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(legend_elements),
        fontsize=44,
        frameon=False
    )
    
    plt.axhline(y=0, color='black', linewidth=1.5, linestyle = 'dashed')
    plt.tight_layout()
    plt.savefig(f'/regimeCapture_{peak_type}.png', dpi=350)
    plt.show()


###########
# Algo Analysis
##############


def plot_metric_cdfs_by_algo(dict_of_dfs, scaling, color_dict, algo_name=None):
    """
    Plot CDFs of performance metrics by algorithm.
    Each algorithm gets its own figure (4 subplots: r, NSE, nBIAS, 1-sigma)
    with a single shared legend on top — one entry per orbit/color,
    showing median and n values.
    """
    metricList = ['r', 'NSE', 'nBIAS', '1-sigma']
    all_summary_rows = []

    # Determine which algos to plot
    if algo_name is None:
        algos_to_plot = set()
        for df in dict_of_dfs.values():
            algos_to_plot.update(df['algo'].unique())
        algos_to_plot = sorted(algos_to_plot)
    else:
        algos_to_plot = [algo_name]

    for algo in algos_to_plot:
        print(f"\n=== Starting plots for algorithm: {algo} ===")
        if algo in ['gauge_swot_match']:
            continue

        n_metrics = len(metricList)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols

        # increased figure height
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(20 * n_cols, 18 * n_rows),
            sharey=True
        )
        axes = axes.flatten()

        summary_rows = []
        orbit_seen = {}  # orbit_label → (line_handle, color, legend_label)

        for i, metric in enumerate(metricList):
            ax = axes[i]

            for orbit_label, df in dict_of_dfs.items():
                if orbit_label in ['Continuous-FSO', 'Continuous-SO']:
                    continue

                df_algo = df[df['algo'] == algo]
                if df_algo.empty:
                    continue

                x_data = df_algo[metric].dropna()
                if x_data.empty:
                    continue

                if metric == 'nBIAS':
                    x_data = x_data.abs()
                
                n_reaches = df_algo[df_algo[metric].notna()]['reach_id'].nunique()
                # Skip if not enough reaches
                if n_reaches <= 20:
                    print(f"  Skipping {orbit_label} for {metric}: n_reaches={n_reaches} <= 20")
                    continue
                    
                x = np.sort(x_data)
                y = np.arange(1, len(x) + 1) / float(len(x))
                
                #n_reaches = df_algo[df_algo[metric].notna()]['reach_id'].nunique()
                median_val = np.round(np.median(x_data), 2)
                #unique_reaches = df_algo['reach_id'].nunique()
                percentile_67 = np.round(np.percentile(x_data, 67), 2)
                obs_per_reach = int(df_algo.groupby('reach_id')[metric].count().median())

                color = color_dict.get(orbit_label, None)

                sns.lineplot(
                    x=x, y=y, ax=ax, color=color, linewidth=12, errorbar=None,
                    label=None
                )

                # Only one legend entry per orbit
                if orbit_label not in orbit_seen:
                    line = ax.lines[-1]
                    legend_label = f"{orbit_label} (n={n_reaches})"
                    orbit_seen[orbit_label] = (line, color, legend_label)

                summary_rows.append({
                    'Algorithm': algo,
                    'Metric': metric,
                    'Orbit': orbit_label,
                    'Median': median_val,
                    '67th Percentile': percentile_67,
                    'n_reach_id': n_reaches,
                    'obs/reach': obs_per_reach
                })

            algo_name_to_use = 'neobam' if algo == 'geobam' else algo

            # Axis formatting — increased font sizes
            ax.set_xlabel(f'{algo_name_to_use}: {metric if metric != "r" else "Pearson r"}', fontsize=64)
            ax.set_ylabel('Proportion', fontsize=64)
            ax.grid(True, which='both', linestyle='--', linewidth=1.5)
            ax.axhline(y=0.50, color='black', linestyle='--', linewidth=8)
            ax.axhline(y=0.67, color='black', linestyle='--', linewidth=8)
            ax.tick_params(axis='both', which='major', labelsize=58, width=2.5, length=12)

            # Metric-specific x-axis limits
            if metric == 'nBIAS':
                ax.set_xlim(-1.5, 1.5)
                ax.set_xlabel(f'{algo_name_to_use}: |nBIAS|')
            elif metric in ['rRMSE', 'nRMSE']:
                ax.set_xlim(0, 2)
            elif metric == 'KGE':
                ax.set_xlim(-1.5, 1.0)
            elif metric == 'NSE':
                ax.set_xlim(-1.5, 1.0)
            elif metric == 'r':
                ax.set_xlim(-1.5, 1.5)
            elif metric == 'RMSE':
                ax.set_xlim(-0.01, 600)
            elif metric == '1-sigma':
                ax.set_xlim(-1.5, 1.5)

        # Hide unused axes
        for j in range(len(axes)):
            if j >= n_metrics:
                axes[j].axis('off')

        # Shared legend — one entry per orbit
        handles, labels = zip(*[(v[0], v[2]) for v in orbit_seen.values()])
        fig.legend(
            handles=handles,
            labels=labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.12),   # more space for large legend
            ncol=len(labels),
            fontsize=66,
            title=f'{algo_name_to_use}',
            title_fontsize=70,
            frameon=True,
            framealpha=0.9,
            handlelength=4.5
        )

        # Increased top space for large legend
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        save_dir = '//figs'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'algoGaugeCDF_{algo}.png')
        print(f"Saving to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
        plt.show()

        all_summary_rows.extend(summary_rows)

    return pd.DataFrame(all_summary_rows)



def plot_metric_cdfs_faceted_by_orbit(dict_of_dfs, scaling, color_dict_algo, algo_name=None):
    """
    Plot CDFs of performance metrics faceted by orbit.
    Each figure corresponds to one metric.
    Each facet (subplot) corresponds to an orbit.
    Lines in each facet show different algorithms with n = unique reach_id count.
    """
    metricList = ['r', 'NSE', 'nBIAS', '1-sigma', 'KGE', 'RMSE', 'nRMSE', 'rRMSE']

    records = []
    for orbit_label, df in dict_of_dfs.items():
        if orbit_label in ['Continuous-FSO', 'Continuous-SO']:
            continue
        df_copy = df.copy()
        df_copy['orbit'] = orbit_label
        if algo_name:
            df_copy = df_copy[df_copy['algo'] == algo_name]
        for metric in metricList:
            if metric not in df_copy.columns:
                continue
            for algo in df_copy['algo'].unique():
                df_algo = df_copy[df_copy['algo'] == algo]
                vals = df_algo[metric].dropna()
                if vals.empty:
                    continue
                if metric == 'nBIAS':
                    vals = vals.abs()
                for val, reach in zip(vals, df_algo.loc[vals.index, 'reach_id']):
                    records.append({
                        'Metric': metric,
                        'Value': val,
                        'algo': algo,
                        'orbit': orbit_label,
                        'reach_id': reach
                    })

    combined_df = pd.DataFrame(records)
    if combined_df.empty:
        print("No data to plot.")
        return

    for metric in metricList:
        df_metric = combined_df[combined_df['Metric'] == metric]
        if df_metric.empty:
            continue

        g = sns.FacetGrid(df_metric, col='orbit', col_wrap=2, height=6, sharex=False, sharey=True)

        def cdf_plot(data, color=None, **kwargs):
            ax = plt.gca()
            for algo in data['algo'].unique():
                df_algo = data[data['algo'] == algo]
                # Only plot if there are at least 10 unique reaches
                unique_reaches = df_algo['reach_id'].nunique()
                if unique_reaches >= 10:
                    x = np.sort(df_algo['Value'])
                    y = np.arange(1, len(x) + 1) / float(len(x))
                    color = color_dict_algo.get(algo, 'gray')
                    ax.plot(x, y, label=algo, color=color, linewidth=4)

        g.map_dataframe(cdf_plot)

        # Now add legends per axis with n counts
        for ax, orbit_label in zip(g.axes.flatten(), g.col_names):
            df_orbit = df_metric[df_metric['orbit'] == orbit_label]
            algo_counts = df_orbit.groupby('algo')['reach_id'].nunique()
            
            # Filter to only include algos with at least 10 unique reaches
            algo_counts = algo_counts[algo_counts >= 10]

            handles = []
            labels = []
            for algo in sorted(algo_counts.index):
                color = color_dict_algo.get(algo, 'gray')
                handles.append(plt.Line2D([0], [0], color=color, lw=3))
                labels.append(f"{algo} (n={algo_counts[algo]})")
            ax.legend(handles, labels, title='Algorithm (Reach Count)', title_fontsize=18, loc='upper left', fontsize=18)
            
            # Make orbit title bold
            ax.set_title(ax.get_title(), fontweight='bold', fontsize=20)

        # Determine which axes are in the bottom row
        num_orbits = len(g.col_names)
        num_cols = 2  # col_wrap=2
        num_rows = int(np.ceil(num_orbits / num_cols))
        bottom_row_start = (num_rows - 1) * num_cols
        
        for idx, ax in enumerate(g.axes.flatten()):
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2)
            ax.set_ylabel('Proportion', fontsize=28)
            ax.tick_params(axis='both', which='major', labelsize=24)
            
            # Check if this is a bottom row axis
            is_bottom_row = idx >= bottom_row_start
            
            if metric == 'nBIAS':
                ax.set_xlim(-1.5, 1.5)
                if is_bottom_row:
                    ax.set_xlabel('|nBIAS|', fontsize=28)
            elif metric in ['rRMSE', 'nRMSE']:
                ax.set_xlim(0, 2)
                if is_bottom_row:
                    ax.set_xlabel(metric, fontsize=28)
            elif metric == 'KGE':
                ax.set_xlim(-1.5, 1.0)
                if is_bottom_row:
                    ax.set_xlabel(metric, fontsize=28)
            elif metric == 'NSE':
                ax.set_xlim(-1.5, 1.5)
                if is_bottom_row:
                    ax.set_xlabel(metric, fontsize=28)
            elif metric == 'r':
                ax.set_xlim(-1.0, 1.1)
                if is_bottom_row:
                    ax.set_xlabel(f'Pearson {metric}', fontsize=28)
            elif metric == 'RMSE':
                ax.set_xlim(-0.01, 600)
                if is_bottom_row:
                    ax.set_xlabel(metric, fontsize=28)
            elif metric == '1-sigma':
                ax.set_xlim(-1.5, 1.5)
                if is_bottom_row:
                    ax.set_xlabel(metric, fontsize=28)
            else:
                if is_bottom_row:
                    ax.set_xlabel(metric, fontsize=28)

        plt.subplots_adjust(top=0.9)
        #plt.suptitle(f'CDF of {metric} across Orbits ({scaling})', fontsize=20, weight='bold')
        plt.savefig(f'//figs/algoMetric_byOrbit_{metric}.png', bbox_inches='tight')        
        plt.show()
        
    return combined_df
