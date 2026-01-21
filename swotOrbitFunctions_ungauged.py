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


#### REVISIT TIME PLOTS
def get_season_orbits(date):
    month = date.month
    day = date.day
    if (month == 3 and day >= 30) or (month in [4, 5, 6]) or (month == 7 and day < 12):
        return '4-7'
    elif (month == 7 and day >= 12) or (month in [7, 8]) or (month == 9 and day < 30):
        return '7-10'
    elif (month == 9 and day >= 30) or (month in [10, 11]):
        return '10-1'
    else:
        return '1-4'

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
            # if df_name in ['Continuous-FSO', 'Continuous-SO']:
            #     continue
            
            if df_name in ['Continuous', 'Science', 'Fast', 'Sampled']:
                df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
                # keep only this season
                df = df[df['season'] == season]
                if season == '4-7':
                    df = df[df['time'] <  pd.Timestamp('2025-03-30', tz="UTC")]
                    
            # Filter consensus only
            df_cons = df[df['algo'] == 'consensus'].copy()
            df_cons = df_cons.sort_values(['reach_id', 'time'])
            print(season, df_name, df_cons.reach_id.nunique(), df_cons.reach_id.value_counts().mean())

            revisit_periods_reach = []

            # Compute revisit period per reach
            for reach, group in df_cons.groupby('reach_id'):
                times = pd.to_datetime(group['time']).sort_values()
                
                if df_name in ['Continuous', 'Science']: 
                    if len(times) >= 10:
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
                            p_lat = group['p_lat'].iloc[0]
                            continent_id = group['continent_id'].iloc[0]
                            xtrk_dist = group['xtrk_dist'].mean()
                            p_width = group['p_width'].iloc[0]
                            scatter_data.append({
                                'df_name': df_name,
                                'reach_id': reach,
                                'revisit_period': revisit_period,
                                'mean_flow': mean_flow,
                                'p_lat': abs(p_lat),
                                'continent_id': continent_id,
                                'xtrk_dist': xtrk_dist,
                                'p_width': p_width,
                                
                            })
                            revisit_periods_reach.append(revisit_period)
                
                if df_name in ['Sampled', 'Fast']: 
                    if len(times) >= 10:
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
                            p_lat = group['p_lat'].iloc[0]
                            continent_id = group['continent_id'].iloc[0]
                            xtrk_dist = group['xtrk_dist'].mean()
                            p_width = group['p_width'].iloc[0]
                            scatter_data.append({
                                'df_name': df_name,
                                'reach_id': reach,
                                'revisit_period': revisit_period,
                                'mean_flow': mean_flow,
                                'p_lat': abs(p_lat),
                                'continent_id': continent_id,
                                'xtrk_dist': xtrk_dist,
                                'p_width': p_width

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
                group['p_lat'], 
                s=10,
                label=f"{df_name} (n={n_reaches})", 
                color=color_dict_timeseries.get(df_name, 'black'),
                alpha=0.5
            )

        #axes[0].set_yscale('log')
        axes[0].set_xlabel('Revisit Period (days)', fontsize=40)
        axes[0].set_ylabel('Latitude', fontsize=40)
        axes[0].set_title(f'Revisit Period vs Lat per Reach — {season}', fontsize=40)
        axes[0].tick_params(axis='both', labelsize=36)
        legend = axes[0].legend(fontsize=30, markerscale=3, handlelength=3.5, loc='best')
        for lh in legend.legendHandles:
            lh._sizes = [200]
        axes[0].grid(True)

        # CDF plot
        for df_name, sorted_rp in cdf_data.items():
            n_reaches = reach_counts.get(df_name, 0)
            cdf = np.arange(1, len(sorted_rp)+1) / len(sorted_rp)
            axes[1].plot(
                sorted_rp, 
                cdf, 
                label=f"{df_name} (n={n_reaches})", 
                color=color_dict_timeseries.get(df_name, 'black'),linewidth=4
            )
        axes[1].axhline(y=0.50, color='black', linestyle='dashed')
        axes[1].set_xlabel('Revisit Period (days)', fontsize=40)
        axes[1].set_ylabel('Proportion', fontsize=40)
        axes[1].set_title(f'CDF of Revisit Period per Orbit — {season}', fontsize=42)
        axes[1].tick_params(axis='both', labelsize=36)
        axes[1].legend(fontsize=36)
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'/figs/revisit_{season}.png', dpi=350)
        plt.show()

def get_revisit_times_overall(dfs_q, color_dict_timeseries):
    # Prepare a list to collect results
    scatter_data = []
    cdf_data = {}
    reach_counts = {}  # store n per df_name

    for df_name, df in dfs_q.items():
        df = df.copy()
        # if df_name in ['Continuous-FSO', 'Continuous-SO']:
        #     continue

        if df_name in ['Continuous', 'Science', 'Fast', 'Sampled']:
            df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

        # Filter consensus only
        df_cons = df[df['algo'] == 'consensus'].copy()
        df_cons = df_cons.sort_values(['reach_id', 'time'])
        #print(df_name, df_cons.reach_id.nunique())

        revisit_periods_reach = []

        # Compute revisit period per reach
        for reach, group in df_cons.groupby('reach_id'):
            times = pd.to_datetime(group['time']).sort_values()

            if df_name in ['Continuous', 'Science']:
                if len(times) > 10:
                    delta_days = times.diff().dt.total_seconds().dropna() / 86400.0
                    revisit_period = delta_days.mean()
                    mean_flow = group['Q'].mean()
                    p_lat = group['p_lat'].iloc[0]

                    scatter_data.append({
                        'df_name': df_name,
                        'reach_id': reach,
                        'revisit_period': revisit_period,
                        'mean_flow': mean_flow,
                        'p_lat': abs(p_lat)
                    })
                    revisit_periods_reach.append(revisit_period)

            elif df_name in ['Fast', 'Sampled']:
                if len(times) > 10:
                    delta_days = times.diff().dt.total_seconds().dropna() / 86400.0
                    revisit_period = delta_days.mean()
                    mean_flow = group['Q'].mean()
                    p_lat = group['p_lat'].iloc[0]
                    scatter_data.append({
                        'df_name': df_name,
                        'reach_id': reach,
                        'revisit_period': revisit_period,
                        'mean_flow': mean_flow,
                        'p_lat': abs(p_lat)
                    })
                    revisit_periods_reach.append(revisit_period)

        if revisit_periods_reach:
            cdf_data[df_name] = np.sort(revisit_periods_reach)
            reach_counts[df_name] = len(revisit_periods_reach)

    # Convert to DataFrame for scatter plot
    scatter_df = pd.DataFrame(scatter_data)
    

    if not scatter_df.empty:
        print(scatter_df.groupby(['df_name']).revisit_period.describe())
    else:
        print("no data")

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    # Scatter plot
    for df_name, group in scatter_df.groupby('df_name'):
        n_reaches = reach_counts.get(df_name, 0)
        axes[0].scatter(
            group['revisit_period'], 
            group['p_lat'], 
            s=10,
            label=f"{df_name} (n={n_reaches})", 
            color=color_dict_timeseries.get(df_name, 'black'),
            alpha=0.5
        )

    #axes[0].set_yscale('log')
    axes[0].set_xlabel('Revisit Period (days)')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Revisit Period vs Mean Flow per Reach — Overall')
    legend = axes[0].legend(fontsize=30, markerscale=3, handlelength=3.5, loc='best')
    for lh in legend.legendHandles:
        lh._sizes = [200]
    axes[0].grid(True)

    # CDF plot
    for df_name, sorted_rp in cdf_data.items():
        n_reaches = reach_counts.get(df_name, 0)
        cdf = np.arange(1, len(sorted_rp)+1) / len(sorted_rp)
        axes[1].plot(
            sorted_rp, 
            cdf, 
            label=f"{df_name} (n={n_reaches})", 
            color=color_dict_timeseries.get(df_name, 'black'), linewidth=4
        )
    axes[1].axhline(y=0.50, color='black', linestyle='dashed')
    axes[1].set_xlabel('Revisit Period (days)')
    axes[1].set_ylabel('Proportion')
    axes[1].set_title('CDF of Revisit Period per Orbit — Overall')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


    
####
# Hydrographs

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

    # Step 2: Find common reach_ids across all labels
    consensus_sets = [
        set(dfs_dict[label][dfs_dict[label]['algo'] == algo]['reach_id'].unique())
        for label in labels if label != 'gauge'
    ]
    common_reach_ids = list(sorted(set.intersection(*consensus_sets).intersection(reach_ids_before_divide)))
    print(f"Found {len(common_reach_ids)} common reach IDs")

    # Define plotting order and sizes
    plot_order = ['gauge', 'Continuous', 'Science', 'Fast', 'Sampled']
    point_sizes = {
        'gauge': 250,        # largest
        'Continuous': 240,   # slightly smaller
        'Science': 150,      # smaller
        'Fast': 150,         # same as Science
        'Sampled': 240        # smallest
    }

    # Step 3: Plot subset of reach_ids
    for reach_id in common_reach_ids[800:900]:
        plt.figure(figsize=(25, 10))
        max_Q_all = []

        # --- Build master sequential axis from gauge data in Continuous ---
        df_continuous_data = dfs_dict.get('Continuous')
        df_gauge_reach = None
        
        if df_continuous_data is not None:
            df_gauge_reach = df_continuous_data[
                (df_continuous_data['reach_id'] == reach_id) &
                (df_continuous_data['algo'] == 'gauge')
            ].copy()

        # Initialize date_to_idx
        date_to_idx = None
        all_dates = None
        divide_idx = 0

        if df_gauge_reach is not None and not df_gauge_reach.empty:
            df_gauge_reach['time'] = pd.to_datetime(df_gauge_reach['time'])
            df_gauge_reach = df_gauge_reach.sort_values('time')
            all_dates = df_gauge_reach['time'].drop_duplicates().reset_index(drop=True)
            master_idx = range(len(all_dates))
            date_to_idx = dict(zip(all_dates, master_idx))
            
            df_gauge_reach['seq_idx'] = df_gauge_reach['time'].map(date_to_idx)
            df_gauge_reach = df_gauge_reach.dropna(subset=['seq_idx'])

            if len(df_gauge_reach['Q'].values) >= 10:
                plt.plot(df_gauge_reach['seq_idx'].values, df_gauge_reach['Q'].values,
                            label="GAUGE", color=color_dict.get('gauge', 'black'),
                             alpha=0.5, linewidth=4, zorder=1)
                max_Q_all.append(df_gauge_reach['Q'].max())
        else:
            # If no gauge data, create sequential index from Continuous consensus data
            df_continuous_consensus = df_continuous_data[
                (df_continuous_data['reach_id'] == reach_id) &
                (df_continuous_data['algo'] == algo)
            ].copy() if df_continuous_data is not None else pd.DataFrame()
            
            if not df_continuous_consensus.empty:
                df_continuous_consensus['time'] = pd.to_datetime(df_continuous_consensus['time'])
                df_continuous_consensus = df_continuous_consensus.sort_values('time')
                all_dates = df_continuous_consensus['time'].drop_duplicates().reset_index(drop=True)
                master_idx = range(len(all_dates))
                date_to_idx = dict(zip(all_dates, master_idx))
            else:
                # Skip this reach if no data at all
                plt.close()
                continue

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

        # Skip plotting if no data was plotted
        if len(max_Q_all) == 0:
            plt.close()
            continue

        # --- Plot vertical divide line + orbit labels ---
        if all_dates is not None:
            divide_idx = (all_dates < divide_date).sum()
        
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

        if (reach_id) == int(73114000751):
            plt.legend(loc='upper right', fontsize=30)
        
        plt.tight_layout()
        plt.savefig(f'/figs/hydrograph_{reach_id}.pdf', bbox_inches='tight', dpi=300)

        plt.show()

        
### REACH
def plot_seasonal_log_ratio_boxplots_by_continent_reach(dfs_dict, consensus_algo, plot=True):
    from scipy.stats import ttest_1samp, shapiro, wilcoxon
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from itertools import combinations
    
    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'Siberia', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'Arctic', '9': 'GR'
    }

    custom_palette = {
        "AF": '#e377c2', "EU": '#2ca02c', "Siberia": '#17becf',
        "AS": '#d62728', "OC": '#9467bd', "SA": 'gold',
        "NA": '#ff7f0e', "Arctic": '#1f77b4'
    }

    continent_order = ['Arctic', 'EU', 'NA', 'Siberia', 'AS','OC', 'AF', 'SA']

    full_season_order = ['4-7', '7-10', '10-1', '1-4']

    for name1, name2 in combinations(dfs_dict.keys(), 2):
        if name1 in ['Continuous']:
            continue


        df1 = dfs_dict[name1]
        df2 = dfs_dict[name2]

        df1_c = df1[df1['algo'] == consensus_algo].copy()
        df2_c = df2[df2['algo'] == consensus_algo].copy()

        df1_c['time'] = pd.to_datetime(df1_c['time'])
        df2_c['time'] = pd.to_datetime(df2_c['time'])

        merged = pd.merge(
            df1_c[['time', 'reach_id', 'Q']],
            df2_c[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=(f"_{name1}", f"_{name2}")
        )
        merged = merged[(merged[f"Q_{name1}"] > 0) & (merged[f"Q_{name2}"] > 0)]
        if merged.empty:
            continue

        merged['log_ratio'] = np.log10(merged[f"Q_{name1}"]) - np.log10(merged[f"Q_{name2}"])
        merged['continent'] = merged['reach_id'].astype(str).str[0]
        merged['continent_name'] = merged['continent'].map(continent_map)
        merged['season'] = merged['time'].apply(get_season_orbits)
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])

        # ---- Reach-level aggregation ----
        reach_medians = merged.groupby(["reach_id", "season", "continent_name"])["log_ratio"].median().reset_index()

        filtered_records = []
        group_pvals = {}
        group_counts = {}

        for (season, continent), group_df in reach_medians.groupby(['season', 'continent_name']):
            log_vals = group_df['log_ratio'].dropna()
            if len(log_vals) < 2 or continent == 'GR':
                continue

            log_vals = modified_z_filter(log_vals)
            if len(log_vals) < 2:
                continue

            group_counts[(season, continent)] = len(log_vals)

            try:
                shapiro_p = shapiro(log_vals)[1]
                if shapiro_p > 0.05:
                    stat, pval = ttest_1samp(log_vals, popmean=0)
                else:
                    stat, pval = wilcoxon(log_vals, alternative='two-sided')
            except Exception:
                pval = np.nan

            group_pvals[(season, continent)] = pval

            filtered_records.extend([
                {'season': season, 'continent_name': continent, 'log_ratio': val}
                for val in log_vals
            ])

        if not filtered_records:
            continue

        filtered_df = pd.DataFrame(filtered_records)

        # Drop sparse groups
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: group_counts.get((row['season'], row['continent_name']), 0) > 10, axis=1)
        ]
        if filtered_df.empty:
            continue

        sample_sizes = reach_medians.groupby("continent_name")["reach_id"].nunique().to_dict()

        filtered_df["Continent (n)"] = filtered_df["continent_name"].apply(
            lambda x: f"{x} (n_reach={sample_sizes.get(x, 0)})"
        )

        # --- Hue order & palette ---
        hue_order = [f"{cont} (n_reach={sample_sizes.get(cont,0)})" 
                     for cont in continent_order if cont in sample_sizes]

        annotated_palette = {label: custom_palette[label.split(' ')[0]] for label in hue_order}

                # prune season_order to only those with data
        season_order = [s for s in full_season_order if s in filtered_df['season'].unique()]
        if not season_order:
            continue

        if plot:
            plt.figure(figsize=(30, 12))
            ax = sns.boxplot(
                data=filtered_df,
                x='season',
                y='log_ratio',
                hue='Continent (n)',
                order=season_order,         
                palette=annotated_palette,
                hue_order=hue_order
            )

            # Log-diff reference lines
            x_pos = len(season_order) - 0.4
            for val, label in zip([0, np.log10(2), np.log10(3), np.log10(5),
                                   -np.log10(2), -np.log10(3), -np.log10(5)],
                                  ['1x', '2x', '3x', '5x', '2x', '3x', '5x']):
                plt.axhline(val, linestyle='--', color='gray', linewidth=3, alpha=0.8)
                plt.text(x_pos-0.09, val, label, color='gray', fontsize=30, va='bottom')

               # --- Add significance stars with fixed order ---
            spread = 0.10
            offset = -0.35
            for season_idx, season in enumerate(season_order):
                for continent_idx, continent in enumerate(continent_order):
                    if (season, continent) not in group_counts:
                        continue
                    n = group_counts[(season, continent)]
                    if n <= 10:
                        continue

                    x_pos = season_idx + offset + continent_idx * spread
                    pval = group_pvals.get((season, continent), None)
                    if pval is not None and pval < 0.05:
                        continent_label = f"{continent} (n_reach={sample_sizes.get(continent,0)})"
                        color = annotated_palette.get(continent_label, 'black')
                        ax.text(x_pos, 0.8, '*', ha='center', va='bottom', fontsize=30, color=color, weight='bold')
            
            
            plt.axhline(0.0, color='k', linestyle='--')
            plt.title(f"{name1} / {name2} Seasonal Discharge Comparison", fontsize=50, pad=20)
            plt.xlabel("Season", fontsize=45)
            plt.ylabel(f"{name1} / {name2} (log)", fontsize=45)
            plt.xticks(fontsize=40)
            plt.yticks(fontsize=40)
            plt.legend(title='Continent', fontsize=34, title_fontsize=38, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylim([-1.0, 1.0])
            plt.tight_layout()
            plt.savefig(f'/figs/ungauged_cons_comparison_{name1}_{name2}.png', dpi=350)
            plt.show()
            
            
            
# COMBO CONTINTENTAL DISCHARGE COMPARISON

def plot_seasonal_log_ratio_by_continent_reach_gauged_vs_ungauged(
    dfs_dict_gauged, dfs_dict_ungauged, consensus_algo, output_dir, plot=True
):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations
    from scipy.stats import ttest_1samp, shapiro, wilcoxon
    name1 = 'Science'
    name2 = 'Continuous'
    # ---- Helpers ----
    def prepare_log_ratio_df(dfs_dict, consensus_algo, label):
        """Compute seasonal log10(Q_Science / Q_Continuous) by reach."""

        df_sci = dfs_dict[name1]
        df_cont = dfs_dict[name2]

        df_sci = df_sci[df_sci['algo'] == consensus_algo].copy()
        df_cont = df_cont[df_cont['algo'] == consensus_algo].copy()

        df_sci['time'] = pd.to_datetime(df_sci['time'])
        df_cont['time'] = pd.to_datetime(df_cont['time'])

        merged = pd.merge(
            df_sci[['time', 'reach_id', 'Q']],
            df_cont[['time', 'reach_id', 'Q']],
            on=['time', 'reach_id'],
            suffixes=('_Science', '_Continuous')
        )

        merged = merged[(merged['Q_Science'] > 0) & (merged['Q_Continuous'] > 0)]
        if merged.empty:
            print('empty')
            return pd.DataFrame()

        merged['log_ratio'] = np.log10(merged['Q_Science']) - np.log10(merged['Q_Continuous'])
        merged['continent'] = merged['reach_id'].astype(str).str[0]
        merged['continent_name'] = merged['continent'].map(continent_map)
        merged['season'] = merged['time'].apply(get_season_orbits)
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])

        # Aggregate by reach/season
        reach_medians = (
            merged.groupby(['reach_id', 'season', 'continent_name'])['log_ratio']
            .median().reset_index()
        )
        reach_medians['group'] = label
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

    # ---- Setup ----
    os.makedirs(output_dir, exist_ok=True)

    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'Siberia', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'Arctic', '9': 'GR'
    }

    custom_palette = {
        "AF": '#e377c2', "EU": '#2ca02c', "Siberia": '#17becf',
        "AS": '#d62728', "OC": '#9467bd', "SA": 'gold',
        "NA": '#ff7f0e', "Arctic": '#1f77b4'
    }
    # custom_palette = {
    #     "AF": 'dimgrey', "EU": 'dimgrey', "Siberia": 'dimgrey',
    #     "AS": 'dimgrey', "OC": 'dimgrey', "SA": 'dimgrey',
    #     "NA": 'dimgrey', "Arctic": 'dimgrey'
    # }
    continent_order = ["AF", "EU", "Siberia", "AS",
                       "Arctic", "NA", "SA", "OC"]

    full_season_order = ['4-7', '7-10', '10-1', '1-4']

    # ---- Prepare data ----
    df_g = prepare_log_ratio_df(dfs_dict_gauged, consensus_algo, label='Gauged')
    df_u = prepare_log_ratio_df(dfs_dict_ungauged, consensus_algo, label='Ungauged')

    combined = pd.concat([df_g, df_u], ignore_index=True)
    
    if combined.empty:
        print("No valid data found for Science/Continuous comparison.")
        return
    
    sample_size_records = []
    
    # ---- Loop over continents ----
    for cont in continent_order:
        df_cont = combined[combined['continent_name'] == cont].copy()
        if df_cont.empty:
            print('cont is empty')
            continue

        group_pvals, group_counts = significance_tests(df_cont)

        # Drop sparse groups
        df_cont = df_cont[df_cont.apply(
            lambda row: group_counts.get((row['season'], row['continent_name'], row['group']), 0) > 10,
            axis=1
        )]
        if df_cont.empty:
            print('season addition is empty')
            continue

        season_order = [s for s in full_season_order if s in df_cont['season'].unique()]
        if not season_order:
            print('no season order')
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

        # Collect sample sizes per season/group
        for (season, group), n in df_cont.groupby(['season', 'group'])['reach_id'].nunique().items():
            sample_size_records.append({
                'continent': cont,
                'season': season,
                'group': group,
                'n_reaches': n
            })

        
        # ---- Plot ----
        if plot:
            import matplotlib.colors as mcolors

            def lighten_color(color, amount=0.5):
                """Lighten the given color by mixing it with white."""
                try:
                    c = mcolors.cnames[color]
                except KeyError:
                    c = color
                c = mcolors.to_rgb(c)
                return tuple(1 - (1 - x) * (1 - amount) for x in c)

            base_color = custom_palette.get(cont, '#333333')
            lighter_color = lighten_color(base_color, amount=0.5)  # lighter for gauged
            palette = {'Gauged': lighter_color, 'Ungauged': base_color}

            plt.figure(figsize=(12, 10))
            ax = sns.boxplot(
                data=df_cont,
                x='season',
                y='log_ratio',
                hue='group',
                order=season_order,
                palette=palette
            )

            # reference lines
            plt.axhline(0, color='k', linestyle='--', linewidth=2)
            for val, label in zip([np.log10(2), np.log10(3), np.log10(5),
                                   -np.log10(2), -np.log10(3), -np.log10(5)],
                                  ['2x', '3x', '5x', '2x', '3x', '5x']):
                plt.axhline(val, linestyle='--', color='gray', linewidth=1, alpha=0.6)
                plt.text(ax.get_xlim()[1] - 0.2, val, label, color='gray', fontsize=25, va='bottom')

            # significance stars — colored by which group deviates more from 0
            for season in season_order:
                for group in ['Gauged', 'Ungauged']:
                    n = group_counts.get((season, cont, group), 0)
                    if n > 10:
                        p = group_pvals.get((season, cont, group), None)
                        if p is not None and p < 0.05:
                            xpos = season_order.index(season)
                            star_color = palette[group]  # lighter if Gauged, darker if Ungauged
                            ax.text(
                                xpos - 0.1 if group == 'Gauged' else xpos + 0.1,
                                ax.get_ylim()[1] * 0.85,
                                '*',
                                ha='center',
                                va='bottom',
                                fontsize=30,
                                color=star_color,
                                fontweight='bold'
                            )


            # ---- Legend with sample sizes ----
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
            plt.ylabel("Science / Continuous (log)", fontsize=32)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.tight_layout()

            outpath = os.path.join(output_dir, f"Seasonal_LogRatio_{cont}_{name1}_{name2}.png")
            plt.savefig(outpath, dpi=500)
            plt.show()
            plt.close()
            print(f"Saved {outpath}")
        
        sample_sizes_df = pd.DataFrame(sample_size_records)
        print(sample_sizes_df)

        
        
# REGIME CHARACTERISTICS

def summarize_overall_Q(dfs_q, algo):
    """
    Summarize log10(Q) and raw Q statistics per reach_id from a dictionary of DataFrames.
    Includes various percentiles, match dates/counts, and normalized IQR.

    Parameters:
        dfs_q (dict): Dictionary of DataFrames keyed by run label

    Returns:
        dict: Dictionary of summary DataFrames per run, including full and split versions
    """
    output = {}
    split_date = pd.to_datetime('2023-07-11')

    def process(df, run_label):
        df = df.copy()
        df = df[df['algo'] == algo]
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'Q', 'reach_id'])
        df = df[df['Q'] > 0]  # Required for log10
        df['logQ'] = np.log10(df['Q'])

        summaries = []
        for reach_id, group in df.groupby('reach_id'):
            logQ_vals = group['logQ'].values
            Q_vals = group['Q'].values
            time_vals = group['time'].values
            count = len(logQ_vals)
            if count == 0:
                continue

            logQ_p5 = np.percentile(logQ_vals, 5)
            logQ_p95 = np.percentile(logQ_vals, 95)
            p5_match = logQ_vals <= logQ_p5
            p95_match = logQ_vals >= logQ_p95

            Q_p20 = np.percentile(Q_vals, 20)
            Q_p40 = np.percentile(Q_vals, 40)
            Q_p60 = np.percentile(Q_vals, 60)
            Q_p80 = np.percentile(Q_vals, 80)
            Q_p25 = np.percentile(Q_vals, 25)
            Q_p75 = np.percentile(Q_vals, 75)
            Q_median = np.median(Q_vals)
            Q_nIQR = (Q_p75 - Q_p25) / Q_median if Q_median != 0 else np.nan

            summary = {
                'reach_id': reach_id,
                'Q_log10_p5': logQ_p5,
                'Q_log10_p95': logQ_p95,
                'Q_log10_mean': np.mean(logQ_vals),
                'Q_log10_median': np.median(logQ_vals),
                'Q_log10_sd': np.std(logQ_vals),
                'Q_log10_IQR': (np.percentile(logQ_vals, 75) - np.percentile(logQ_vals, 25)) if count >= 10 else np.nan,
                'Q_log10_cv': (np.std(logQ_vals) / np.mean(logQ_vals)) if count >= 10 and np.mean(logQ_vals) != 0 else np.nan,
                'Q_count': count,
                'Q_sum': np.sum(Q_vals),
                'Q_nIQR': Q_nIQR,
                'Q_p20': Q_p20,
                'Q_p40': Q_p40,
                'Q_p60': Q_p60,
                'Q_p80': Q_p80,
                'Q_log10_p5_date': time_vals[p5_match][0] if np.any(p5_match) else pd.NaT,
                'Q_log10_p5_n': np.sum(p5_match),
                'Q_log10_p95_date': time_vals[p95_match][0] if np.any(p95_match) else pd.NaT,
                'Q_log10_p95_n': np.sum(p95_match),
            }

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        summary_df['run'] = run_label
        summary_df['algo'] = algo

        return summary_df

    for run_label, df in dfs_q.items():
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # Always process full version
        output[run_label] = process(df, run_label)
        print('done', run_label)

        # Split Continuous and Sampled into fast and science periods
#         if run_label in ['Continuous', 'Sampled']:
#             df_fast = df[df['time'] < split_date]
#             df_science = df[df['time'] >= split_date]

#             run_fast = f"{run_label}-fast"
#             run_science = f"{run_label}-science"

#             output[run_fast] = process(df_fast, run_fast)
#             print('done', run_fast)

#             output[run_science] = process(df_science, run_science)
#             print('done', run_science)

    return output


# Algo by orbit
def plot_logQ_diff_grouped_by_orbit_combo_with_stats(
    dfs_dict, algos, orbit_pairs, color_dict=None
):
    """
    Plot reach-level boxplots of log10(Q_orbit1 / Q_orbit2) for SAME algorithms
    across specified orbit pairs, with statistical testing and significance stars
    colored by algorithm. Sample size capped at 5000 for stats.
    """
    all_data = []

    # Collect reach-level log ratios
    for orbit1, orbit2 in orbit_pairs:
        for algo in algos:
            df1 = dfs_dict.get(orbit1)
            df2 = dfs_dict.get(orbit2)
            if df1 is None or df2 is None:
                continue

            df1_a = df1[df1['algo'] == algo].dropna(subset=['time', 'Q']).copy()
            df2_a = df2[df2['algo'] == algo].dropna(subset=['time', 'Q']).copy()
            if df1_a.empty or df2_a.empty:
                continue

            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(df1_a['time']):
                df1_a['time'] = pd.to_datetime(df1_a['time'], errors='coerce')
            if not pd.api.types.is_datetime64_any_dtype(df2_a['time']):
                df2_a['time'] = pd.to_datetime(df2_a['time'], errors='coerce')

            # Merge on reach and time
            merged = pd.merge(
                df1_a[['time', 'reach_id', 'Q']],
                df2_a[['time', 'reach_id', 'Q']],
                on=['time', 'reach_id'],
                suffixes=(f"_{orbit1}", f"_{orbit2}")
            )

            merged = merged[(merged[f"Q_{orbit1}"] > 0) & (merged[f"Q_{orbit2}"] > 0)]
            if merged.empty:
                continue

            merged['LogQRatio'] = np.log10(merged[f"Q_{orbit1}"]) - np.log10(merged[f"Q_{orbit2}"])

            # Compute reach-level median
            reach_medians = merged.groupby('reach_id')['LogQRatio'].median().reset_index()
            reach_medians['Algorithm'] = algo
            reach_medians['OrbitCombo'] = f"{orbit1} / {orbit2}"

            all_data.append(reach_medians)

    if not all_data:
        print("No valid data found for the given algorithms and orbit pairs.")
        return

    df_plot = pd.concat(all_data, ignore_index=True)

    # Stat calculations for annotations
    group_pvals = {}
    group_counts = {}

    for (orbit_combo, algo), group_df in df_plot.groupby(['OrbitCombo', 'Algorithm']):
        vals = group_df['LogQRatio'].values
        if len(vals) < 5:
            continue

        vals_filtered = modified_z_filter(vals)
        if len(vals_filtered) < 5:
            continue

        # Cap at 5000 for stats
        if len(vals_filtered) > 5000:
            vals_filtered = np.random.choice(vals_filtered, 4500, replace=False)

        group_counts[(orbit_combo, algo)] = len(vals_filtered)

        try:
            shapiro_p = shapiro(vals_filtered)[1]
            if shapiro_p > 0.05:
                stat, pval = ttest_1samp(vals_filtered, popmean=0)
            else:
                stat, pval = wilcoxon(vals_filtered, alternative='two-sided')
        except Exception:
            pval = np.nan

        group_pvals[(orbit_combo, algo)] = pval

    # Plot boxplot (reach-level)
    plt.figure(figsize=(14, 7))

    # Map legend directly to algorithm names
    df_plot['AlgorithmLegend'] = df_plot['Algorithm']

    # Use the original color_dict
    ax = sns.boxplot(
        data=df_plot,
        x='OrbitCombo',
        y='LogQRatio',
        hue='AlgorithmLegend',
        palette=color_dict,
        showfliers=False,
    )
    # Set alpha for all boxes
    for patch in ax.patches:
        patch.set_alpha(0.9)
        
    # Reference horizontal lines
    for val, label in zip([0, np.log10(2), np.log10(3), np.log10(5),
                           -np.log10(2), -np.log10(3), -np.log10(5)],
                          ['1x', '2x', '3x', '5x', '2x', '3x', '5x']):
        plt.axhline(val, linestyle='--', color='gray', linewidth=1, alpha=0.4)
        plt.text(len(df_plot['OrbitCombo'].unique()) - 0.5, val, label,
                 va='bottom', ha='right', fontsize=16, color='gray')

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.ylabel('log10(Run1 / Run2)')
    #plt.xlabel('Run Combincation')

    plt.title('Algorithm Discharge Comparison', fontsize=28)
    #plt.xticks(ha='right') #rotation=30, 
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add significance stars colored by algorithm
    orbit_order = list(df_plot['OrbitCombo'].unique())
    algo_order = list(df_plot['Algorithm'].unique())
    n_hues = len(algo_order)
    spread = 0.18

    for i, orbit_combo in enumerate(orbit_order):
        for j, algo in enumerate(algo_order):
            key = (orbit_combo, algo)
            pval = group_pvals.get(key, None)
            offset = (j - (n_hues - 1) / 2) * spread
            xpos = i + offset
            if pval is not None and pval < 0.05:
                color = color_dict.get(algo, 'black')
                ax.text(xpos, 0.5, '*', ha='center', va='bottom', fontsize=30, color=color, weight='bold')

    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/figs/algo_q_diff_ungauged_poster.png', bbox_inches='tight', dpi=400)

    plt.show()
    
    
    
############
# Reach Comparison - Q Distribution
##############

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
    
    for reach_id in all_reaches[1000:1020]:
        plt.figure(figsize=(12, 8))
        data_dict = {}
        
        # --- Loop over runs/orbits ---
        for run_label, df in df_dict.items():
            # if run_label in ['Continuous-SO']: 
            #     continue
                
            df_reach = df[(df['reach_id'] == reach_id) & (df['algo'] == algo)]
            if df_reach.empty:
                continue
            
            data = df_reach[variable].dropna().to_numpy()
            if len(data) < 10:
                continue
            
            data = np.sort(data)
            cdf = np.arange(1, len(data) + 1) / len(data)
            
            color = color_dict.get(run_label, "grey")
            plt.plot(data, cdf, label=f"{run_label} (n={len(data)})", color=color, linewidth=2)
            data_dict[run_label] = data
            
            # Overlay gauge ONLY for Continuous
            if run_label == "Continuous":
                df_gauge = df[(df['reach_id'] == reach_id) & (df['algo'] == "gauge")]
                if not df_gauge.empty:
                    data = df_gauge[variable].dropna().to_numpy()
                    if len(data) >= 10:
                        data = np.sort(data)
                        cdf = np.arange(1, len(data) + 1) / len(data)
                        plt.plot(
                            data, cdf, 
                            label=f"{run_label} (gauge, n={len(data)})",
                            color="black", linewidth=2, linestyle="--"
                        )
                        # Store under a combined key
                        data_dict[f"{run_label}_gauge"] = data            
            
        # --- Require Fast, Science, and Continuous ---
        if not required_runs.issubset(data_dict.keys()):
            plt.close()
            continue      
        
        # --- Compute pairwise distances ---
        run_labels = list(data_dict.keys())
        dist_text = ""
        
        # Define allowed pairs (unordered)
        allowed_pairs = {
            frozenset(["Fast", "Continuous-FSO"]),
            frozenset(["Science", "Continuous"]),
            frozenset(["Science", "Continuous-SO"]),
            frozenset(["Sampled", "Fast"]),
            frozenset(["Fast", "Continuous"]),
            frozenset(["Science", "Sampled"]),
        }
        
        for i in range(len(run_labels)):
            for j in range(i + 1, len(run_labels)):
                a1, a2 = run_labels[i], run_labels[j]
                
                # Skip if not an allowed pair
                if frozenset([a1, a2]) not in allowed_pairs:
                    continue

                d1, d2 = data_dict[a1], data_dict[a2]

                # Only compute stats if both samples have n >= 10
                if len(d1) < 10 or len(d2) < 10:
                    continue  

                ks_stat, p_value = ks_2samp(d1, d2)
                combined = np.sort(np.unique(np.concatenate([d1, d2])))
                cdf1 = np.searchsorted(d1, combined, side='right') / len(d1)
                cdf2 = np.searchsorted(d2, combined, side='right') / len(d2)
                l1_dist = np.sum(np.abs(cdf1 - cdf2) * 
                                 np.diff(np.concatenate([[combined[0]-1], combined])))
                emd = wasserstein_distance(d1, d2)

                dist_text += (
                    f"{a1}-{a2} KS/p_value={ks_stat:.2f}/{p_value:.2f}, L1={l1_dist:.2f}\n"
                )

                summary_rows.append({
                    'reach_id': reach_id,
                    'pair': f"{a1}-{a2}",
                    'KS': ks_stat,
                    'p_value': p_value,
                    'L1': l1_dist,
                    'EMD': emd
                })
        
        # --- Final formatting ---
        plt.title(f"Consensus {variable} FDC — Reach {reach_id}", fontsize=18)
        plt.xlabel(variable, fontsize=16)
        plt.ylabel("CDF", fontsize=16)
        plt.grid(True)
        plt.legend(loc="lower right", fontsize=12)
        
        # Put distances in a text box
        if dist_text:
            plt.gcf().text(
                0.5, -0.05, dist_text,
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black')
            )
        
        plt.tight_layout()
        plt.show()
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    return summary_df



def plot_reach_consensus_cdfs(df_dict, variable, algo, color_dict):
    summary_rows = []
    required_runs = {"Fast", "Science", "Continuous"}
    all_reaches = sorted(set().union(*[df['reach_id'].unique() for df in df_dict.values()]))

    for reach_id in tqdm(all_reaches, desc="Processing reaches"):
        #plt.figure(figsize=(12, 8))
        data_dict = {}

        # Prepare reach data for each run
        for run_label, df in df_dict.items():
            #if run_label in ['Continuous-SO']:
            #    continue

            df_reach = df[(df['reach_id'] == reach_id) & (df['algo'] == algo)].copy()
            if df_reach.empty:
                continue

            df_reach['time'] = pd.to_datetime(df_reach['time'])
            data = df_reach[variable].dropna().to_numpy()
            if len(data) < 10:
                continue

            # Plot CDF
            # data_sorted = np.sort(data)
            # cdf = np.arange(1, len(data_sorted)+1)/len(data_sorted)
            # color = color_dict.get(run_label, "grey")
            # plt.plot(data_sorted, cdf, label=f"{run_label} (n={len(data_sorted)})", color=color, linewidth=2)

            data_dict[run_label] = df_reach[['time', variable, 'CV']].copy()

            # Overlay gauge for Continuous
            if run_label == "Continuous":
                df_gauge = df[(df['reach_id']==reach_id) & (df['algo']=="gauge")].copy()
                if not df_gauge.empty:
                    df_gauge['time'] = pd.to_datetime(df_gauge['time'])
                    data_sorted = np.sort(df_gauge[variable].dropna())
                    cdf = np.arange(1, len(data_sorted)+1)/len(data_sorted)
                    plt.plot(data_sorted, cdf,
                             label=f"{run_label} (gauge, n={len(data_sorted)})",
                             color="black", linewidth=2, linestyle="--")
                    data_dict[f"{run_label}_gauge"] = df_gauge[['time', variable, 'CV']].copy()

        if not required_runs.issubset(data_dict.keys()):
            plt.close()
            continue

        run_labels = list(data_dict.keys())
        dist_text = ""
        allowed_pairs = {
            frozenset(["Fast", "Continuous-FSO"]),
            frozenset(["Science", "Continuous-SO"]),
            frozenset(["Science", "Continuous"]),
            frozenset(["Sampled", "Fast"]),
            frozenset(["Fast", "Continuous"]),
            frozenset(["Science", "Sampled"]),
        }

        for i in range(len(run_labels)):
            for j in range(i+1, len(run_labels)):
                a1, a2 = run_labels[i], run_labels[j]
                if frozenset([a1, a2]) not in allowed_pairs:
                    continue

                df1, df2 = data_dict[a1], data_dict[a2]

                # Merge by time for point-to-point
                merged = pd.merge(df1, df2, on='time', suffixes=(f'_{a1}', f'_{a2}'))
                if len(merged) < 10:
                    continue

                Q1 = merged[f'{variable}_{a1}'].to_numpy()
                Q2 = merged[f'{variable}_{a2}'].to_numpy()

                # Point-to-point log differences
                log_diffs = np.log10(Q1) - np.log10(Q2)
                n_points = len(log_diffs)
                median_logdiff = np.median(log_diffs)
                try:
                    if n_points >= 10:
                        if np.all(log_diffs == log_diffs[0]):
                            logdiff_pval = 1.0  # all identical, non-significant
                        else:
                            shapiro_p = shapiro(log_diffs)[1]
                            if shapiro_p > 0.05:
                                _, logdiff_pval = ttest_1samp(log_diffs, popmean=0)
                            else:
                                _, logdiff_pval = wilcoxon(log_diffs)
                except:
                    logdiff_pval = np.nan

                # Full distribution metrics on matched points
                abs_diff = Q1 - Q2
                log_ratio = np.log10(Q1 / Q2 + 1e-12)
                try:
                    norm_res = abs_diff - np.mean(abs_diff)
                except:
                    norm_res = np.array([np.nan])
                
                # --- Distribution metrics (full distribution, ≥20 points each) ---
                Q1_full = df1[variable].dropna().to_numpy()
                Q2_full = df2[variable].dropna().to_numpy()
                if len(Q1_full) >= 20 and len(Q2_full) >= 20:
                    ks_stat, p_value = ks_2samp(Q1_full, Q2_full)
                    combined = np.sort(np.unique(np.concatenate([Q1_full, Q2_full])))
                    cdf1 = np.searchsorted(Q1_full, combined, side='right') / len(Q1_full)
                    cdf2 = np.searchsorted(Q2_full, combined, side='right') / len(Q2_full)
                    l1_dist = np.sum(np.abs(cdf1 - cdf2) * np.diff(np.concatenate([[combined[0]-1], combined])))
                    emd = wasserstein_distance(Q1_full, Q2_full)
                else:
                    ks_stat = p_value = l1_dist = emd = np.nan
                    
                dist_text += f"{a1}-{a2} KS/p={ks_stat:.2f}/{p_value:.2f}, L1={l1_dist:.2f}\n"

                row = {
                    'reach_id': reach_id,
                    'pair': f"{a1}-{a2}",
                    'KS': round(ks_stat, 2),
                    'p_value': round(p_value, 2),
                    'L1': round(l1_dist, 2),
                    'EMD': round(emd, 2),
                    'n_points': n_points,
                    'median_logdiff': round(median_logdiff, 2),
                    'logdiff_p_value': round(logdiff_pval, 2) if not np.isnan(logdiff_pval) else np.nan,
                    'med_bias': round(np.median(abs_diff), 2),
                    'nBIAS': round((np.sum(Q1 - Q2)/len(Q2)) / np.mean(Q2), 2) if np.mean(Q2)!=0 else np.nan,
                    '|nBIAS|': round(abs((np.sum(Q1 - Q2)/len(Q2))/np.mean(Q2)), 2) if np.mean(Q2)!=0 else np.nan,
                    'med_log10_ratio': round(np.median(log_ratio), 2),
                    'RMSE': round(np.sqrt(mean_squared_error(Q1, Q2)), 2),
                    'nRMSE': round(np.sqrt(mean_squared_error(Q1, Q2)) / (Q1.max()-Q1.min()), 2),
                    'NSE': round(1 - (np.sum((Q2 - Q1)**2) / np.sum((Q2 - np.mean(Q2))**2)), 2) if np.sum((Q2 - np.mean(Q2))**2)!=0 else np.nan,
                    'Pearson_r': round(np.corrcoef(Q1, Q2)[0,1], 2),
                    'sigE': round(np.nanpercentile(norm_res, 67), 2) if isinstance(norm_res, (pd.Series, np.ndarray)) else np.nan,
                    'Q1_std': round(np.std(Q1), 2),
                    'Q2_std': round(np.std(Q2), 2),
                    'Q1_CV': round(merged[f'CV_{a1}'].iloc[0], 2) if f'CV_{a1}' in merged.columns else np.nan,
                    'Q2_CV': round(merged[f'CV_{a2}'].iloc[0], 2) if f'CV_{a2}' in merged.columns else np.nan
                }
                
                summary_rows.append(row)
                
        # plt.title(f"Consensus {variable} FDC — Reach {reach_id}", fontsize=18)
        # plt.xlabel(variable, fontsize=16)
        # plt.ylabel("CDF", fontsize=16)
        # plt.grid(True)
        # plt.legend(loc="lower right", fontsize=12)
        # if dist_text:
        #     plt.gcf().text(0.5, -0.05, dist_text,
        #                    ha='center', va='top', fontsize=12,
        #                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black'))
        # plt.tight_layout()
        # plt.show()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


#################
# Summary metrics
################

def plot_pair_stacked_bars(df, metric):
    # Define bins and labels
    bins = [-np.inf, -0.3, 0, 0.3, np.inf]
    labels = ["<-0.3", "-0.3 to 0.0", "0.0 to 0.3", ">0.3"]

    def categorize(row):
        if row["p_value"] > 0.5:
            return "p>0.5"
        elif row["p_value"] < 0.05:
            return pd.cut([row[metric]], bins=bins, labels=labels)[0]
        else:
            return None

    df["category"] = df.apply(categorize, axis=1)
    df = df.dropna(subset=["category"])

    # Keep only the relevant pairs
    df = df[df['pair'].isin(['Fast-Continuous-FSO', 'Science-Continuous-SO'])]

    counts = df.groupby(["pair", "category"]).size().unstack(fill_value=0)
    ordered_cols = ["p>0.5"] + labels
    counts = counts.reindex(columns=ordered_cols, fill_value=0)

    pair_color_maps = {
        "Fast-Continuous-FSO": {
            "p>0.5": "lightgrey",
            "<-0.3": "#D83A34",
            "-0.3 to 0.0": "lightcoral",
            "0.0 to 0.3": "#FFD100",
            ">0.3": "#FD8500"
        },
        "Science-Continuous-SO": {
            "p>0.5": "lightgrey",
            "<-0.3": "#D83A34",
            "-0.3 to 0.0": "lightcoral",
            "0.0 to 0.3": "#2B9EB3",
            ">0.3": "blue"
        },
    }

    x_labels = {
        "Fast-Continuous-FSO": "Fast-Continuous",
        "Science-Continuous-SO": "Science-Continuous"
    }

    fig, ax = plt.subplots(figsize=(18, 20))
    x_positions = np.arange(len(counts.index)) * 1.5
    bar_width = 0.6

    # Plot bars
    for i, pair in enumerate(counts.index):
        pair_counts = counts.loc[pair]
        colors = [pair_color_maps[pair][c] for c in counts.columns]
        bottom = 0
        for col, color in zip(counts.columns, colors):
            val = pair_counts[col]
            ax.bar(x_positions[i], val, bottom=bottom, color=color, width=bar_width, edgecolor='black')
            if val > 0:
                ax.text(x_positions[i]+0.05, bottom+val/2, f"{(val/pair_counts.sum())*100:.1f}%", 
                        ha='center', va='center', fontsize=34)
            bottom += val
        ax.text(x_positions[i], pair_counts.sum()+100, str(pair_counts.sum()), 
                ha='center', va='bottom', fontsize=38, fontweight='bold')

        # Create small legend manually above each bar
        for j, col in enumerate(counts.columns):
            ax.add_patch(plt.Rectangle((x_positions[i]-0.4 + j*0.15, pair_counts.sum()+650), 0.12, 150, 
                                       facecolor=colors[j], edgecolor='black'))
            ax.text(x_positions[i]-0.40 + j*0.15, pair_counts.sum()+1000, col, fontsize=28, rotation=30)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([x_labels[p] for p in counts.index], fontsize=38)
    ax.set_ylabel("Count", fontsize=40)
    ax.set_xlabel("Run Comparison", fontsize=40)
    ax.tick_params(axis='y', labelsize=38)
    ax.set_ylim([0, max(counts.sum(axis=1))*1.35])

    plt.savefig(f'/figs/globalMapMetric_{metric}.png', dpi=350)
    plt.show()
    
    

def plot_pair_by_continent(df, pairs_to_plot, metric):
    # Continent mapping
    continent_map = {
        '1': 'AF', '2': 'EU', '3': 'SI', '4': 'AS',
        '5': 'OC', '6': 'SA', '7': 'NA', '8': 'AR', '9': 'GR'
    }

    # nBIAS bins and labels
    bins = [-np.inf, -0.3, 0, 0.3, np.inf]
    labels = ["<-0.3", "-0.3 to 0.0", "0.0 to 0.3",
            ">0.3"]


    # Assign categories
    def categorize(row):
        if row["p_value"] > 0.5:
            return "p>0.5"
        elif row["p_value"] < 0.05:
            return pd.cut([row[metric]], bins=bins, labels=labels)[0]
        else:
            return None

    df["category"] = df.apply(categorize, axis=1)
    df = df.dropna(subset=["category"])

    # Map continents
    df.loc[:, "continent"] = df["reach_id"].astype(str).str[0].map(continent_map)
    df = df.dropna(subset=["continent"])
    df = df[df['continent'] != 'GR']
    # Keep only requested pairs
    df = df[df["pair"].isin(pairs_to_plot)]
    
    

    # One plot per orbit pair
    for pair in pairs_to_plot:
        if pair in ["Fast-Continuous-FSO", "Fast-Continuous"]:
            cmap = {
                "<-0.3": "#D83A34",
                "-0.3 to 0.0": "lightcoral",
                "0.0 to 0.3": "#FFD100",
                ">0.3": "#FD8500"
            }
            grey_color = "lightgrey"
        else:
            cmap = {
                "<-0.3": "#D83A34",
                "-0.3 to 0.0": "lightcoral",
                "0.0 to 0.3": "#2B9EB3",
                ">0.3": "blue"
            }
            grey_color = "lightgrey"
        subdf = df[df["pair"] == pair]

        # Group counts by continent + category
        counts = subdf.groupby(["continent", "category"]).size().unstack(fill_value=0)

        # Reorder stacked bar categories
        ordered_cols = ["p>0.5"] + labels
        counts = counts.reindex(columns=ordered_cols, fill_value=0)

        # Colors
        colors = [grey_color] + [cmap[l] for l in labels]

        # Plot — make taller and leave room on right
        fig, ax = plt.subplots(figsize=(20, 20))  # much taller figure
        counts.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.2)
        
        pair_labels = {
        "Fast-Continuous-FSO": "Fast-Continuous",
        "Science-Continuous-SO": "Science-Continuous"
            }
        # Annotate % inside stacks
        for i, cont in enumerate(counts.index):
            total = counts.loc[cont].sum()
            cumulative = 0
            for j, col in enumerate(counts.columns):
                value = counts.loc[cont, col]
                if value > 0:
                    y = cumulative + value / 2.1
                    percent = (value / total) * 100
                    ax.text(i + 0.4, y, f"{percent:.1f}%", ha="center", va="center", fontsize=24, color="black")
                cumulative += value
            # Total above bar
            ax.text(i, total + 1, str(total), ha="center", va="bottom", fontsize=28, fontweight="bold")

        ax.set_ylabel("Count", fontsize=35)
        ax.set_xlabel("Continent", fontsize=35)
        ax.set_title(f"Significant Q {metric} by Continent ({pair_labels[pair]})", fontsize=30)
        ax.tick_params(axis='x', rotation=90, labelsize=33)
        ax.legend(title="Category", loc="upper left", fontsize=24)  # bbox_to_anchor=(1.25, 1), push legend farther right
        ax.tick_params(axis='y', labelsize=38)
        ax.set_ylim([0, 2100])
        ax.set_xlim([-0.5, len(counts.index) - 0.5 + 1.0])  # extra 1.0 adds space to the right
        # Add generous margins
       #plt.subplots_adjust(right=2.8, left=0.1, top=0.93, bottom=0.15)
        plt.savefig(f'/figs/globalMapMetric_continent_{metric}_{pair}.png', dpi=350)

        plt.show()    
