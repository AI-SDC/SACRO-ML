'''
worst_case_attack.py

Runs a worst case attack based upon predictive probabilities stored in two .csv files
'''

import argparse
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from attacks import metrics
from attacks import report

logging.basicConfig(level=logging.INFO)

P_THRESH = 0.05


def generate_array(n_rows, beta):
    '''
    Generate a single array of predictions. Picks a class for
    each observation and then picks a probability of chat class from a beta distribution.
    '''
    preds = np.zeros((n_rows, 2), float)
    for n in range(n_rows):
        train_class = np.random.choice(2)
        train_prob = np.random.beta(1, beta)
        preds[n, train_class] = train_prob
        preds[n, 1 - train_class] = 1 - train_prob
    return preds

def generate_arrays(n_rows_in, n_rows_out, train_beta, test_beta):
    '''
    Generate train and test prediction arrays
    '''
    train_preds = generate_array(n_rows_in, train_beta)
    test_preds = generate_array(n_rows_out, test_beta)
    return train_preds, test_preds

def make_dummy_data(args):
    '''Makes dummy data for testing functionality'''
    logger = logging.getLogger("dummy-data")
    logger.info("Making dummy data with %d rows in and %d out", args.n_rows_in, args.n_rows_out)
    logger.info("Generating rows")
    train_preds, test_preds = generate_arrays(args.n_rows_in, args.n_rows_out, args.train_beta, args.test_beta)
    logger.info("Saving files") 
    np.savetxt("train_preds.csv", train_preds, delimiter=",")
    np.savetxt("test_preds.csv", test_preds, delimiter=",")

def run_attack(args, make_report=True):
    '''
    Wrapper method for attack running.
    '''
    logger = logging.getLogger("run-attack")
    logger.info("Loading train predictions form %s", args.in_sample_filename)
    train_preds = np.loadtxt(args.in_sample_filename, delimiter=",")
    logger.info("Loaded %d rows", len(train_preds))
    logger.info("Loading test predictions form %s", args.in_sample_filename)
    test_preds = np.loadtxt(args.out_sample_filename, delimiter=",")
    logger.info("Loaded %d rows", len(test_preds))
    mia_metrics, metadata = attack(args, train_preds, test_preds)

    # do some baseline attacks
    logger.info("Performing baseline attacks")

    dummy_metrics = []
    dummy_metadata = None
    for _ in range(args.dummy_reps):
        dummy_train, dummy_test = generate_arrays(len(train_preds), len(test_preds), 2, 2)
        temp_dummy_metrics, dummy_metadata = attack(args, dummy_train, dummy_test)
        dummy_metrics += temp_dummy_metrics
    
    if args.dummy_reps > 0:
        metadata['experiment_details']['n_baseline_experiments_done'] = f'Done {args.n_reps} reps for {args.dummy_reps} splits of synthetic data (total = {len(dummy_metrics)})'
    else:
        metadata['experiment_details']['n_baseline_experiments_done'] = "None"

    if make_report:
        pdf = report.create_mia_report(metadata, mia_metrics, dummy_metrics, dummy_metadata)
        pdf.output(f"{args.report_name}.pdf", 'F')
        json_details = report.create_json_report(metadata, mia_metrics, dummy_metrics, dummy_metadata)
        with open(f"{args.report_name}.json", 'w') as json_file:
            json_file.write(json_details)

def get_n_significant(p_val_list, p_thresh, bh_fdr_correction=False):
    '''
    Helper method to determine if values within a list of p-values are significant at
    p_thresh. Can peform multiple testing correction.
    '''
    if not bh_fdr_correction:
        return sum([1 for p in p_val_list if p <= p_thresh])
    else:
        p_val_list = np.asarray(sorted(p_val_list))
        m = len(p_val_list)
        hoch_vals = np.array(
            [(k / m) * P_THRESH for k in range(1, m + 1)]
        )
        bh_sig_list = p_val_list <= hoch_vals
        if any(bh_sig_list):
            n_sig_bh = (np.where(bh_sig_list)[0]).max() + 1
        else:
            n_sig_bh = 0
        return n_sig_bh


    
def attack(args, train_preds, test_preds):
    '''attack
    Runs the attack based upon the predictions in train_preds and test_preds, and params
    in args
    '''
    logger = logging.getLogger("attack")
    logger.info("Sorting probabilities to leave highest value in first column")
    train_preds = -np.sort(-train_preds, axis=1)
    test_preds = -np.sort(-test_preds, axis=1)

    logger.info("Creating MIA data")
    mi_x = np.vstack((train_preds, test_preds))
    mi_y = np.hstack(
        (
            np.ones(len(train_preds)),
            np.zeros(len(test_preds))
        )
    )

    mia_metrics = []
    for rep in range(args.n_reps):
        logger.info("Rep %d of %d", rep, args.n_reps)
        mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(
            mi_x, mi_y, test_size=args.test_prop, stratify=mi_y
        )
        attack_classifier = RandomForestClassifier()
        attack_classifier.fit(mi_train_x, mi_train_y)

        mia_metrics.append(
            metrics.get_metrics(
                attack_classifier,
                mi_test_x,
                mi_test_y
            )
        )

    logger.info("Finished simulating attacks")
    metadata = {
        'global_metrics': {},
        'experiment_details': {
            'n_reps': args.n_reps,
            'n_in_sample': len(train_preds),
            'n_out_sample': len(test_preds),
            'attack model': "Random Forest",
            'in_sample_filename': args.in_sample_filename,
            'out_sample_filename': args.out_sample_filename,
            'p_val_thresh': args.p_thresh
        }
    }

    # Compute NULL AUC
    auc_std = np.sqrt(
        metrics.expected_auc_var(0.5, mi_test_y.sum(), len(mi_test_y) - mi_test_y.sum())
    )
    # Assuming auc is normal, compute the probability of the NULL generating an AUC higher
    # than the NULL
    auc_p_vals = [1 - norm.cdf(m['AUC'], loc=0.5, scale=auc_std) for m in mia_metrics]
    
    metadata['global_metrics']['null_auc_3sd_range'] = f"{0.5 - 3*auc_std:.4f} -> {0.5 + 3*auc_std:.4f}"
    metadata['global_metrics']['n_sig_auc_p_vals'] = get_n_significant(auc_p_vals, args.p_thresh)
    metadata['global_metrics']['n_sig_auc_p_vals_corrected'] = get_n_significant(
        auc_p_vals,
        args.p_thresh,
        bh_fdr_correction=True
    )

    pdif_vals = [np.exp(-m['PDIF01']) for m in mia_metrics]
    metadata['global_metrics']['n_sig_pdif_vals'] = get_n_significant(pdif_vals, args.p_thresh)
    metadata['global_metrics']['n_sig_pdif_vals_corrected'] = get_n_significant(
        pdif_vals,
        args.p_thresh,
        bh_fdr_correction=True
    )

    return mia_metrics, metadata
    

def main():
    '''main method to parse arguments and invoke relevant method'''

    parser = argparse.ArgumentParser(description=(
        'Perform a worst case attack from saved model predictions'
        )
    )
    
    subparsers = parser.add_subparsers()
    dummy_parser = subparsers.add_parser('make-dummy-data')
    dummy_parser.add_argument('--num-rows-in',
        action='store',
        dest='n_rows_in',
        type=int,
        required=False,
        default=100,
        help=(
            'How many rows to generate in the in-sample file. Default = %(default)d'
        )
    )

    dummy_parser.add_argument('--num-rows-out',
        action='store',
        dest='n_rows_out',
        type=int,
        required=False,
        default=100,
        help=(
            'How many rows to generate in the out-of-sample file. Default = %(default)d'
        )
    )

    dummy_parser.add_argument('--train-beta',
        action='store',
        type=float,
        required=False,
        default=5,
        dest="train_beta",
        help=(
            'Value of b parameter for beta distribution used to sample the in-sample probabilities. '
            'High values will give more extreme probabilities. Set this value higher than '
            '--test-beta to see successful attacks. Default = %(default)f'
        )
    )

    dummy_parser.add_argument('--test-beta',
        action='store',
        type=float,
        required=False,
        default=2,
        dest="test_beta",
        help=(
            'Value of b parameter for beta distribution used to sample the out-of-sample '
            'probabilities. High values will give more extreme probabilities. Set this value '
            'lower than --train-beta to see successful attacks. Default = %(default)f'
        )
    )

    dummy_parser.set_defaults(func=make_dummy_data)

    attack_parser = subparsers.add_parser('run-attack')
    attack_parser.add_argument('-i', '--in-sample-preds',
        action='store',
        dest='in_sample_filename',
        required=False,
        type=str,
        default="train_preds.csv",
        help=(
            'csv file containing the predictive probabilities (one column per class) for the '
            'training data (one row per training example). Default = %(default)s'
        )
    )

    attack_parser.add_argument('-o', '--out-of-sample-preds',
        action='store',
        dest='out_sample_filename',
        required=False,
        type=str,
        default="test_preds.csv",
        help=(
            'csv file containing the predictive probabilities (one column per class) for the '
            'non-training data (one row per training example). Default = %(default)s'
        )
    )

    attack_parser.add_argument('-r', '--n-reps',
        type=int,
        required=False,
        default=5,
        action="store",
        dest="n_reps",
        help=(
            'Number of repetitions (splitting data into attack model training and testing '
            'paritions to perform. Default = %(default)d'
        )
    )

    attack_parser.add_argument('-t', '--test-prop',
        type=float,
        required=False,
        default=0.3,
        action="store",
        dest="test_prop",
        help=(
            'Proportion of examples to be used for testing when fiting the attack model. '
            'Default = %(default)f'
        )
    )

    attack_parser.add_argument('--report-name',
        type=str,
        action='store',
        dest='report_name',
        default='mia_report',
        required=False,
        help=(
            'Filename for the report output. Default = %(default)s. Code will append .pdf and .json'
        )
    )

    attack_parser.add_argument('--dummy-reps',
        type=int,
        action='store',
        dest='dummy_reps',
        default=1,
        required=False,
        help=(
            'Number of dummy datasets to sample. Each will be assessed with --n-reps train and '
            'test splits. Set to 0 to do no baseline calculations. Default = %(default)d'
        )
    )

    attack_parser.add_argument('--p-thresh',
        action='store',
        type=float,
        default=P_THRESH,
        required=False,
        dest='p_thresh',
        help=(
            'P-value threshold for significance testing. Default = %(default)f'
        )
    )

    attack_parser.set_defaults(func=run_attack)


    args = parser.parse_args()
    print(args)
    try:
        args.func(args)
    except AttributeError as e:
        print("Invalid command. Try --help to get more details")

if __name__ == '__main__':
    main()