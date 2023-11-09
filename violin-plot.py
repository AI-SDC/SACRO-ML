'''WIP code to create plots for the SACRO experiments
Creates a seaborn facet grid where rows are attack metrics
and columns are structural metrics. Within each plot in the grid there is one
paired violin plot per dataset and each paired violin plot separates the data points
based on the structural risk value'''
# %% Import libraries
import os
import glob
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt

# %% Setup logging and seaborn
sns.set_theme(style="dark")
sns.set(font_scale=2.0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% Parameters to modify
DATA_PATH = '/Users/simonr04/sacro_results' # where are the results .csv files
GLOB_PATTERN = 'DT_*.csv' # which results .csv files to load
# GLOB_PATTERN = 'xgboost_*.csv'
P_THRESH = 0.05 # Threshold for binary p-value significance evaluation
ATTACK_SCENARIO = 'lira' # Which Attack scenario are we analysing

# %% Retrieve the files to load
file_list = glob.glob(
    os.path.join(DATA_PATH, GLOB_PATTERN)
)
logger.info("Found %d files", len(file_list))

# %% Load the files
total_df = None
for filename in file_list:
    logger.info("Reading %s", filename)
    df = pd.read_csv(filename)
    logger.info("Read %d rows from %s", len(df), filename)
    if total_df is None:
        total_df = df
    else:
        total_df = pd.concat((total_df, df), ignore_index=True)
    logger.info("Total rows = %d", len(total_df))

# %% Summarise over repetitions
# Within the .csv files, each experiments is repeated, with an index in the repetitions column
# Some columns are constant over repetitions -- when we summarise these we just keep the first value
# Some columns change and need to be combined. A **mean** is used. TODO: is this best?

# These are the columns that will be combined via **mean**
COLS_TO_AVERAGE = [
    'attack_TPR', 'attack_FPR', 'attack_FAR', 'attack_TNR',
    'attack_PPV', 'attack_NPV', 'attack_FNR',
    'attack_ACC', 'attack_F1score', 'attack_Advantage',
    'attack_AUC', 'attack_P_HIGHER_AUC', 'attack_FMAX01', 'attack_FMIN01',
    'attack_FDIF01', 'attack_PDIF01', 'attack_FMAX02', 'attack_FMIN02',
    'attack_FDIF02', 'attack_PDIF02', 'attack_FMAX001', 'attack_FMIN001',
    'attack_FDIF001', 'attack_PDIF001', 'attack_pred_prob_var',
    'attack_TPR@0.5', 'attack_TPR@0.2', 'attack_TPR@0.1', 'attack_TPR@0.01',
    'attack_TPR@0.001', 'attack_TPR@1e-05', 'attack_n_pos_test_examples',
    'attack_n_neg_test_examples', 'mia_hyp_min_samples_split',
    'mia_hyp_min_samples_leaf', 'mia_hyp_max_depth', 'attack_DoF_risk',
    'attack_k_anonymity_risk', 'attack_class_disclosure_risk',
    'attack_unnecessary_risk', 'attack_lowvals_cd_risk'
]

# These are the columns that are constant (or don't matter), we just take the first value
# If only files from a single classifier are used, then these will be classifier specific
# NOTE: this operation will remove all hyper-parameter columns

CONSTANT_COLS = [
    'dataset',
    'target_classifier',
    'target_clf_file',
    'attack_classifier',
    'attack_clf_file',
    'repetition',
    'target_generalisation_error',
    'param_id',
    'attack_n_pos_test_examples',
    'attack_n_neg_test_examples',
    'mia_hyp_min_samples_split',
    'mia_hyp_min_samples_leaf',
    'mia_hyp_max_depth'
]

# Make an aggregation object and perform the aggregations
aggregations = {}
for col_name in CONSTANT_COLS:
    aggregations[col_name] = 'first' # any agg that can handle chars will work
for col_name in COLS_TO_AVERAGE:
    aggregations[col_name] = 'mean'

agg_df = total_df.groupby(['scenario', 'model_data_param_id']).agg(aggregations)
agg_df.reset_index(inplace=True)

# Check the sizes of the result make sure all aggregation has worked correctly - good for spotting bugs
n_reps_worstcase = len(total_df[total_df['scenario'] == 'WorstCase']['repetition'].unique())
n_reps_lira = len(total_df[total_df['scenario'] == 'lira']['repetition'].unique())
n_reps_structural = len(total_df[total_df['scenario'] == 'Structural']['repetition'].unique())

n_rows_worstcase = len(total_df[total_df['scenario'] == 'WorstCase'])
n_rows_lira = len(total_df[total_df['scenario'] == 'lira'])
n_rows_structural = len(total_df[total_df['scenario'] == 'Structural'])

n_rows_original = len(total_df)
n_rows_predicted = n_rows_worstcase / n_reps_worstcase + \
    n_rows_lira / n_reps_lira + \
    n_rows_structural / n_reps_structural

n_rows_actual = len(agg_df)

assert n_rows_actual == n_rows_predicted

# %% Add any additional columns required
agg_df.loc[agg_df['attack_P_HIGHER_AUC'] < 1e-6, 'attack_P_HIGHER_AUC'] = 1e-6
agg_df['attack_mlogp_auc'] = (-np.log(agg_df['attack_P_HIGHER_AUC']))

# %% Define the metrics of interest

# These are the structural metrics to use (all of them)
structural_metrics = [
    'attack_k_anonymity_risk',
    'attack_DoF_risk',
    'attack_class_disclosure_risk',
    'attack_unnecessary_risk',
    'attack_lowvals_cd_risk'
]

# These are the attack metrics to use (add others / remove / whatever)
attack_metrics = [
    'attack_AUC',
    'attack_mlogp_auc',
    'attack_FDIF01',
    'attack_PDIF01',
    'attack_Advantage'
]

# %%
# Seaborn provides really useful methods for grids of plots
# This code creates a dataframe that can make use of this.
# The dataframe will have one row for every combination of:
# - an attack metric
# - a structural metric
# - a dataset
# - a model_run_param_id
# - a scenario
# I.e. each experiment will now be multiple rows, where each row will allow us to compare one
# pair of attack metric and structural metric.

# Columns that serve to identify an experiment. Note redundant (dataset is in the id), but
# we need dataset for the plotting
unique_cols = ['model_data_param_id', 'dataset']

# Take only rows corresponding to structural scenarios, and just the structural columns
structural_metric_df = agg_df[agg_df['scenario'] == 'Structural'].loc[:, unique_cols + structural_metrics]

# Take only rows corresponding to the attack scenario of interest, and just the attack metric columns
attack_metric_df = agg_df[agg_df['scenario'] == ATTACK_SCENARIO].loc[:, unique_cols + attack_metrics]

# Melt both dfs to9 a long thing version where, instead of one column per metric, they have each
# metric value for each unique experiment on a different row
structural_metric_df = pd.melt(structural_metric_df, id_vars=unique_cols)
attack_metric_df = pd.melt(attack_metric_df, id_vars=unique_cols)

# Rename the columns to avoid a clash and make things clearer
structural_metric_df.rename(columns={'variable': 'structural_metric_name', 'value': 'structural_metric_value'}, inplace=True)
attack_metric_df.rename(columns={'variable': 'attack_metric_name', 'value': 'attack_metric_value'}, inplace=True)

# %%
# Do an outer join to get all combinations
full_df = pd.merge(
    attack_metric_df,
    structural_metric_df,
    how='outer',
    left_on=['model_data_param_id', 'dataset'],
    right_on=['model_data_param_id', 'dataset']
)

# Put the index back as columns
full_df.reset_index(inplace=True)
# %% Make the plot
g = sns.FacetGrid(
    full_df,
    col='structural_metric_name',
    row='attack_metric_name',
    margin_titles=True,
    height=10,
    sharey=False # Means that columns within a row share a y-axis, but different scales can be used on each row
)

g.map(sns.violinplot,'dataset', 'attack_metric_value', hue=full_df['structural_metric_value'], split=True,
        inner="quart", fill=False,
        palette={1: "r", 0: ".35"}, # blue when value is 1, grey otherwise
        linewidth=3
)
g.tick_params('x', rotation=90)


# %% Code to make a scatter plot for one of the violins
# Pick your favourite dataset, structural metric and attack metric
dataset = 'minmax mimic2-iaccd'
structural_metric = 'attack_class_disclosure_risk'
attack_metric = 'attack_Advantage'

plot_cols = ['model_data_param_id', 'structural_metric_value', 'attack_metric_value']
sub_df = full_df.loc[
    (full_df['dataset'] == dataset) &
    (full_df['structural_metric_name'] == structural_metric) &
    (full_df['attack_metric_name'] == attack_metric)
    , plot_cols]
sub_df.rename(columns={'structural_metric_value': structural_metric, 'attack_metric_value': attack_metric}, inplace=True)

# Get the target model generalisation error
temp = agg_df.loc[
    (agg_df['scenario'] == ATTACK_SCENARIO) &
    (agg_df['dataset'] == dataset)
    , ['model_data_param_id', 'target_generalisation_error']]

plot_df = pd.merge(sub_df, temp, how='left', on='model_data_param_id')
sns.set(font_scale=1.0)
sns.scatterplot(
    plot_df,
    x='target_generalisation_error',
    y=attack_metric,
    hue=structural_metric,
    alpha=0.5
)
plt.title(f'Dataset = {dataset}, attack type = {ATTACK_SCENARIO}')
# %%
