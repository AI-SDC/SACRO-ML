# %%
import os
import glob
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_PATH = '/Users/simonr04/sacro_results'
GLOB_PATTERN = 'DT_*.csv'
# %%
file_list = glob.glob(
    os.path.join(DATA_PATH, GLOB_PATTERN)
)
logger.info("Found %d files", len(file_list))
# %%
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
COLS_TO_AVERAGE = [
    'attack_TPR',
    'attack_FPR',
    'attack_FAR',
    'attack_TNR',
    'attack_PPV',
    'attack_NPV',
    'attack_FNR',
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
CONSTANT_COLS = ['dataset',
                 # 'scenario',
                  'target_classifier', 'target_clf_file',
       'attack_classifier', 'attack_clf_file', 'repetition',
       'target_generalisation_error', 
       #'model_data_param_id', 
       'param_id',
       'criterion', 'splitter', 'max_depth', 'min_samples_split',
       'min_samples_leaf', 'max_features', 'class_weight', 'attack_TPR',
    'attack_n_pos_test_examples',
       'attack_n_neg_test_examples', 'mia_hyp_min_samples_split',
       'mia_hyp_min_samples_leaf', 'mia_hyp_max_depth']

aggregations = {}
for col_name in CONSTANT_COLS:
    aggregations[col_name] = 'first' # any agg will work
for col_name in COLS_TO_AVERAGE:
    aggregations[col_name] = 'mean'

agg_df = total_df.groupby(['scenario', 'model_data_param_id']).agg(aggregations).reset_index(inplace=False)

# Check the sizes
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

# %% Example, plot miAUC for a single dataset, stratified by a strucutral risk metric
plot_metric = 'attack_AUC'
structural_metric = 'attack_k_anonymity_risk'
dataset = 'minmax mimic2-iaccd'

# Need a df for
always_cols = ['model_data_param_id']
temp_plot_metric = agg_df[agg_df['scenario'] == 'WorstCase'].loc[:, always_cols + [plot_metric, "dataset"]].set_index('model_data_param_id')
temp_structural_metric = agg_df[agg_df['scenario'] == 'Structural'].loc[:, always_cols + [structural_metric]].set_index('model_data_param_id')
plot_df= temp_plot_metric.join(temp_structural_metric, how='left')
import pylab as plt
plt.scatter(plot_df['attack_AUC'], plot_df['attack_k_anonymity_risk'], c=plot_df['attack_k_anonymity_risk'])
# %%
import seaborn as sns
sns.set_theme(style="dark")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
               split=True, inner="quart", fill=False,
               palette={"Yes": "g", "No": ".35"})
# %%
chart = sns.violinplot(data=plot_df, x="dataset", y="attack_AUC", hue="attack_k_anonymity_risk",
               split=True, inner="quart", fill=False,
               palette={1: "g", 0: ".35"})
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
# %%
