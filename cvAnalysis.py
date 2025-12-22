#### CV ANALYSIS
### Elisa Friedmann
##10/20/2025


'''
Code to implement simple coefficient of variation calculation per algorithm and subsequent threshold removal.
Note it is currently structured to take a dictionary of dataframes due to my orbit analysis manuscript
This could be kept to include multiple devSet runs
'''

################
# FUNCTIONS
################

import datetime
import time
import pathlib
import os,sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
import numpy as np
import pandas as pd
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
from itertools import islice
from scipy.stats import pearsonr
from itertools import combinations
from scipy.stats import genextreme
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
from scipy import stats
from scipy.stats import median_abs_deviation, shapiro, ttest_1samp, wilcoxon


plt.rcParams['font.family'] = 'serif'

plt.rcParams.update({
    'axes.titlesize': 30,    # Title font size
    'axes.labelsize': 22,     # Axis labels font size
    'xtick.labelsize': 20,    # X-axis ticks font size
    'ytick.labelsize': 20,    # Y-axis ticks font size
    'legend.fontsize': 20,    # Legend font size
    'font.size': 20,          # Global font size for text
    'figure.titlesize': 24,   # Figure title font size
    'lines.linewidth': 2,     # Line width
    'axes.linewidth': 2,      # Axis line width
    'axes.grid': True,        # Show grid
    'grid.linestyle': '--',   # Dashed grid lines
    'grid.alpha': 0.5,        # Grid line transparency
    'figure.figsize': (10, 6) # Figure size (width, height in inches)
})

color_dict = {
    'sic4dvar': 'green',
    'momma': 'blue',
    'neobam': 'purple',
    'consensus': 'sienna',
    'metroman': 'orange',
    'geobam': 'purple',
    'hivdi': 'deeppink',
    'sad': 'tomato',
    'gauge': 'dimgrey',
    'gauge_swot_match': 'silver'
}

def calc_cons(df):
    if 'time_str' in df.columns:
        df = df[df['time_str'] != 'no_data']
    if 'time' not in df.columns:
        print('NO TIME COLUMN FOUND')
        
    if 'consensus' not in df['algo'].unique():
            algo_Q_cons_values = (
                df[(df['algo'] != 'gauge') & (df['algo'] != 'gauge_swot_match') ]
                .groupby(['reach_id', 'time'])['Q']
                .median()
                .reset_index()
            )
            algo_Q_cons_values['algo'] = 'consensus'
            df = pd.concat([df, algo_Q_cons_values], ignore_index=True)
    return df


def remove_low_cv_and_recalc_consensus(dfs_dict, CV_thresh):
    """
    For each DataFrame in the input dictionary:
    - Removes rows where CV < threshold
    - Drops existing 'consensus' rows
    - Recalculates consensus and appends it

    Parameters:
    - dfs_dict (dict): Dictionary of DataFrames keyed by label.
    - cv_col (str): Name of the CV column (default ' CV').

    Returns:
    - dict: Dictionary of processed DataFrames.
    """

    cleaned_dict = {}
    for label, df in dfs_dict.items():
        df = df.copy()
        df = df[df['CV'] > CV_thresh]
        df = df[df['algo'] != 'consensus']
        df = calc_cons(df)
        cleaned_dict[label] = df

    return cleaned_dict


def calculate_metrics(df, reaches):
    """
    Calculate performance metrics for all algorithms,
    grouped by 'reach_id' and 'algo' from combined DataFrames.
    Includes data > 1 and less than 1e7
    Will automatically calculate consensus if it does not exist

    Parameters:
    df (pd.DataFrame): DataFrame containing discharge data
    reaches (list): List of unique reach IDs to evaluate

    Returns:
    pd.DataFrame: A DataFrame with calculated metrics for each 'reach_id' and 'algo'
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = df.astype(object)
    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    df = df[(df['Q'] > 1) & (df['Q'] < 1e7)]

    # Add 'consensus' if missing
    if 'consensus' not in df['algo'].unique():
        algo_Q_cons_values = (
            df[(df['algo'] != 'gauge') & (df['algo'] != 'gauge_swot_match') ]
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
                #print('aligned_df empty or len < 10')
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
            sigma1 = np.nanpercentile(norm_res, 68) if isinstance(norm_res, pd.Series) else np.nan
            sigma1 = ((SQ-GQ)/GQ - ((SQ-GQ)/GQ).mean(skipna=True)).abs().quantile(0.68, interpolation="linear")
            res = np.abs((SQ - GQ)) if np.mean(GQ) != 0 else np.nan
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


def coeffVar(df, reach_list):
    """
    Calculate CV for each algo in a list
    Slightly inefficient because it calculates CV a few times 

    Parameters:
    df (pd.DataFrame): DataFrame containing discharge data
    reaches (list): List of unique reach IDs to evaluate

    Returns:
    pd.DataFrame: A DataFrame with calculated metrics for each 'reach_id' and 'algo', with the 
    gauge and consensus CV appended to each row for comparison as needed
    """
    # Filter the DataFrame by the list of reaches
    filtered_df = df[df['reach_id'].isin(reach_list)]

    # Group by 'reach_id' and 'algorithm' and calculate CV
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
    
    # Group by 'reach_id' and 'algorithm' and calculate CV
    grouped_cons = filtered_df[filtered_df['algo'] == 'consensus'].drop_duplicates(subset=['reach_id', 'time', 'Q']).groupby(['reach_id']).agg(
        mean_cons_Q=('Q', 'mean'),
        sd_cons_Q=('Q', 'std')
    ).reset_index()    
    
    grouped_cons['CV_cons'] = np.where(
        grouped_cons['mean_cons_Q'] == 0,
        np.nan,
        grouped_cons['sd_cons_Q'] / grouped_cons['mean_cons_Q']
)

    # Group by 'reach_id' and 'algorithm' and calculate CV
    grouped_gauge = filtered_df[filtered_df['algo'] == 'gauge'].drop_duplicates(subset=['reach_id', 'time', 'Q']).groupby(['reach_id']).agg(
        mean_gauge_Q=('Q', 'mean'),
        sd_gauge_Q=('Q', 'std')
    ).reset_index().dropna()
    
    grouped_gauge['CV_gauge'] = np.where(
        grouped_gauge['mean_gauge_Q'] == 0,
        np.nan,
        grouped_gauge['sd_gauge_Q'] / grouped_gauge['mean_gauge_Q']
)
    
    # Merge with cons CV
    merged_df = pd.merge(grouped, grouped_cons, on='reach_id', how='left')
    
    # Merge with gauge CV
    final_df = pd.merge(merged_df, grouped_gauge, on='reach_id', how='left')
    
    return round(final_df, 3) #grouped, grouped_cons, grouped_gauge


def append_coeffVar(dfs_q):
    for label, df in dfs_q.items():
        if df.empty:
            continue

        # Add coeffvar, make sure to calculate CV only where ocnsensus exists
        coeff_df = coeffVar(df, df[df['algo']=='consensus'].reach_id.unique())

        # Merge CV to orig df
        df = df.merge(
            coeff_df[['reach_id', 'algo', 'CV', 'CV_cons', 'CV_gauge']],
            on=['reach_id', 'algo'], how='left'
        )

        # Update the DataFrame in the dictionary
        dfs_q[label] = df

    return dfs_q

def plot_cdf_coeff(dfs_q, color_dict, algos_to_plot, CV_thresh):
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


        # Plot customization
        plt.hlines(y=0.66, xmin=0, xmax=10, color='black', linestyle='--', linewidth=3)
        plt.xlabel(f'Coefficient of Variation ({label})', fontsize=30)
        plt.ylabel('Proportion', fontsize=30)
        plt.xticks(np.arange(0, 3.6, 0.25), fontsize=25, rotation=45)
        plt.yticks(fontsize=25, rotation=45)
        plt.gca().tick_params(axis='y', pad=15)
        plt.legend(loc='lower right', fontsize=26)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"CV_noThresh_{CV_thresh}.png", dpi=300)
        plt.xlim([0, 2.5])

        plt.show()
        
def plot_metric_cdfs(df, algo, algo_threshold, mnt):
    metric_cols = ['r', 'nBIAS', 'NSE', '1-sigma']
    colors = {'r': 'tab:blue', '1-sigma': 'tab:orange', 'nBIAS': 'tab:green', 'NSE': 'tab:red'}

    # --- Read CV dataset ---
    df_cv_metrics = calculate_metrics(
        df=df[['algo', 'Q', 'time', 'gauge_discharge', 'gauge_time', 'reach_id']],
        reaches=list(df["reach_id"].unique())
    )
    df_cv_algo = df_cv_metrics[df_cv_metrics['algo'] == algo]

    def metric_summary(label, df_metrics, metric):
        vals = df_metrics[metric].abs().dropna() if metric == 'nBIAS' else df_metrics[metric].dropna()
        if vals.empty:
            return None
        return {
            'filter_stage': label,
            'algo': algo,
            'metric': metric,
            'p32': np.percentile(vals, 32).round(2),
            'median': np.median(vals).round(2),
            'p67': np.percentile(vals, 67).round(2),
            'n_reaches': df_metrics.loc[vals.index, 'reach_id'].nunique(),
            'obs/reach': df_metrics.reach_id.value_counts().median()
        }

    summary_rows = []
    summary_info = {}  

    for metric in metric_cols:
        for label, df_metrics in [
            (f'CV > {algo_threshold}', df_cv_algo)
        ]:
            summary = metric_summary(label, df_metrics, metric)
            if summary:
                summary_rows.append(summary)
                summary_info[(metric, label)] = summary

    summary_df = pd.DataFrame(summary_rows)

    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        for stage, df_metrics, style in [
            (f'CV > {algo_threshold}', df_cv_algo, '-')
        ]:
            df_metrics_reach = df_metrics.drop_duplicates(subset='reach_id')
            vals = df_metrics_reach[metric].abs().dropna() if metric == 'nBIAS' else df_metrics_reach[metric].dropna()            

            if vals.empty:
                continue
            x = np.sort(vals)
            y = np.arange(1, len(x) + 1) / len(x)

            summary = summary_info.get((metric, stage), None)
            if summary:
                n_reaches = summary['n_reaches']
                obs_per_reach = summary['obs/reach']
                # Values for ylines
                ylines = [0.32, 0.5, 0.67]
                ytext = [summary['p32'], summary['median'], summary['p67']]
            else:
                n_reaches = np.nan
                obs_per_reach = np.nan
                ylines = []
                ytext = []

            plt.plot(
                x, y, color=colors[metric], linewidth=4, linestyle=style,
                label=f'Version 1, n={n_reaches}, obs/reach={obs_per_reach}'
            )

            # Add horizontal lines with text
            for yl, val in zip(ylines, ytext):
                plt.axhline(y=yl, color='black', linewidth=1.5, linestyle='--')
                # Text slightly above the line
                plt.text(0.95, yl + 0.02, str(val),
                         horizontalalignment='right', fontsize=14, color='black')

        plt.xlim(-0.5, 1)
        plt.ylim(0, 1)
        plt.xlabel(f"{metric}", fontsize=20)
        plt.ylabel("Proportion", fontsize=20)
        plt.title(f"{metric} - Version 1", fontsize=22)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"cdf_{metric}_{algo_threshold}.png", dpi=300)
        plt.show()

    return summary_df[['filter_stage', 'algo', 'metric', 'p32', 'median', 'p67', 'n_reaches', 'obs/reach']]
    

    
    
########
# Inputs
#######

def main():

    mnt = 'relPermML'
    base_path = '/nas/cee-water/cjgleason/ellie/SWOT/confluence/' #where confluence runs are located
    output_path = '/nas/cee-water/cjgleason/ellie/SWOT/orbitMS/data/' #general output location
    output_subdir = 'abcd_perm' #subfolder for results of that run
    
    
    CV_thresh = 0.5

    
    #paths to data
    devset_path = os.path.join(output_path, output_subdir, f'all_gauge_{mnt}.csv')
    devset = pd.read_csv(devset_path)
    devset_fullRange_path = os.path.join(output_path, output_subdir, f'all_gauge_fullRange_{mnt}.csv')
    devset_swot_path = os.path.join(output_path, output_subdir, f'swot_swot_{mnt}.csv')
    
    #read data
    devset = pd.read_csv(devset_path)
    devset_fullRange = pd.read_csv(devset_fullRange_path)
    devset_swot = pd.read_csv(devset_swot_path)

    devset_gauge = {'devSet' : devset}
    print('loaded datasets')


    devset_swot = devset_swot.dropna(subset='time_str').drop(['algo', 'run'], axis=1)
    devset_swot['time'] = pd.to_datetime(devset_swot['time_str'], format='mixed', errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    devset_swot=devset_swot.drop_duplicates(subset=['reach_id', 'time'])


    devset_gauge_swot = {}
    for label, df in devset_gauge.items():
        df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
        df = calc_cons(df)

        #Make sure there are similar gauge/consensus reach_ids
        cons_df = df[df['algo'] == 'consensus'][['reach_id', 'time']]
        gauge_ids = df[df['algo'] == 'gauge']['reach_id'].unique()
        cons_df = cons_df[cons_df['reach_id'].isin(gauge_ids)]
        df = df[
            (df['algo'] != 'gauge') |
            (
                (df['algo'] == 'gauge') &
                (df.set_index(['reach_id', 'time']).index.isin(cons_df.set_index(['reach_id', 'time']).index))
            )
        ]
        
        #Calculate metrics here
        df = calculate_metrics(df=df, reaches=list(df.reach_id.unique()))  


        df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
        print(df.columns.values.tolist(), df.shape, df.algo.unique())
        df_swot = df.merge(devset_swot, on=['reach_id', 'time'], how='left')

        devset_gauge_swot[label] = df_swot 
        print(df_swot.algo.unique(), df_swot.shape)
        print('done: ', label)

        
##########################################      
# Calc and plot coeffVar (no thresholding)
###########################################
    dfs_gauge_swot_metrics = append_coeffVar(dfs_q=devset_gauge_swot)

    algos_to_plot = ['hivdi', 'sic4dvar', 'momma', 'neobam', 'consensus', 'geobam', 'metroman', 'gauge_swot_match','gauge']

    plot_cdf_coeff(dfs_q=dfs_gauge_swot_metrics, color_dict=color_dict, algos_to_plot = algos_to_plot, CV_thresh=CV_thresh)        

####################
# Apply CV Threshold
####################

    dfs_gauge_swot_metrics_recalc = remove_low_cv_and_recalc_consensus(dfs_gauge_swot_metrics, CV_thresh = CV_thresh)
    
    #Drop original CV and recalc based on the remaining data
    for label, df in dfs_gauge_swot_metrics_recalc.items():
        cols_to_drop = ['CV', 'CV_cons', 'CV_gauge', 'RMD_cons']
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols)
        dfs_gauge_swot_metrics_recalc[label] = df  
            
##########################################      
# Calc and plot coeffVar (WITH thresholding)
# Save and plot CV of new thresholded dataset
###########################################

    dfs_gauge_swot_metrics_recalc = append_coeffVar(dfs_q=dfs_gauge_swot_metrics_recalc)
    for label, df in dfs_gauge_swot_metrics_recalc.items():
        df.to_csv(os.path.join(output_path, output_subdir, f'all_q_cv_{CV_thresh}_{mnt}_{label}.csv'), index=False)

    algos_to_plot = ['hivdi', 'sic4dvar', 'momma', 'neobam', 'consensus', 'geobam', 'metroman', 'gauge_swot_match','gauge']

    plot_cdf_coeff(dfs_q=dfs_gauge_swot_metrics_recalc, color_dict=color_dict, algos_to_plot = algos_to_plot, CV_thresh=CV_thresh)

    
#############################################   
# Calc and plot metrics from CV Thresholding
#############################################   
    
    summary_df = plot_metric_cdfs(
        df=dfs_gauge_swot_metrics_recalc['devSet'],
        algo='consensus',
        algo_threshold=0.5,
        mnt='relPermML'
    )

    print(summary_df)
    
if __name__ == "__main__":
    main()