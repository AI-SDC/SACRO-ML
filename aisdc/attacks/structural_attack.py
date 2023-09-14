"""
structural_attack.py.

Runs a number of 'static' structural attacks,based on:
- the target model's properties
- the TREs risk appetite as applied to tables and standard regressions
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import numpy as np

#tree-based model types currently supported
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from acro import ACRO

from aisdc import metrics
from aisdc.attacks import report
from aisdc.attacks.attack import Attack
from aisdc.attacks.attack_report_formatter import GenerateJSONModule
from aisdc.attacks.failfast import FailFast
from aisdc.attacks.target import Target

#import aisdc.safemodel.classifiers.safedecisiontreeclassifier as safedt
#import aisdc.safemodel.classifiers.saferandomforestclassifier as saferf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("structural_attack")


TREE_BASED_MODELS = ['DecisionTreeClassifier','RandomForestClassifier','XGBClassifier']



def unnecessary_risk(model:BaseEstimator,target_type:str)->Bool:
    """
    Checks whether a model's hyper-parameters against
    a set of rules that predict the top 20% most risky.
    
    This check is designed to assess whether a model is
    likely to be **unnecessarily** risky, i.e.,
    whether it is highly likely that a different combination of hyper-parameters
    would have led to model with similar or better accuracy on the task
    but with lower membership inference risk.
    
    The rules applied from an experimental study using a grid search in which:
    - max_features was one-hot encoded from the set [None, log2, sqrt]
    - splitter was encoded using 0=best, 1=random
    
    The target models created were then subject to membership inference attacks (MIA) and the 
    hyper-param combinations rank-ordered according to MIA AUC.
    Then a decision tree trained to recognise whether hyper-params combintions were in the 20% most risky.
    The rules below were extracted from that tree for the 'least risky' nodes
   
    """
    # Returns 1 if high risk, otherwise 0
    
    #all three types support max_depth
    max_depth = float(model.max_depth) if model.max_depth  else 500
    if target_type in ['rf','xgboost']:
        n_estimators = model.n_estimators
    if target_type in ['dt','rf']:
        max_features = model.max_features
        min_samples_leaf = model.min_samples_leaf
        min_samples_split = model.min_samples_split
        
    
    if target_type == "rf":
        max_features= model.max_features
        
        if max_depth > 3.5 and n_estimators > 35 and max_features != None:
            return 1
        if max_depth > 3.5 and n_estimators > 35 and min_samples_split <= 15 and max_features == None and model.bootstrap:
            return 1
        if max_depth > 7.5 and 15 < n_estimators <= 35 and min_samples_leaf <= 15 and not model.bootstrap:
            return 1
    elif target_type == "dt":
        splitter = model.splitter
        if max_depth > 7.5 and min_samples_leaf <= 7.5 and min_samples_split <= 15:
            return 1
        if splitter == "best" and max_depth > 7.5 and min_samples_leaf <= 7.5 and min_samples_split > 15:
            return 1
        if splitter == "best" and max_depth > 7.5 and 7.5 < min_samples_leaf <= 15 and max_features == None:
            return 1
        if splitter == "best" and 3.5 < max_depth <= 7.5 and max_features == None and min_samples_leaf <= 7.5:
            return 1
        if splitter == "random" and max_depth > 7.5 and min_samples_leaf <= 7.5 and max_features == None:
            return 1
    elif target_type == "xgboost":
        
        if max_depth > 3.5 and 3.5 < n_estimators <= 12.5 and model.min_child_weight <= 1.5:
            return 1
        if max_depth > 3.5 and n_estimators > 12.5 and model.min_child_weight <= 3:
            return 1
        if max_depth > 3.5 and n_estimators > 62.5 and 3 < model.min_child_weight <= 6:
            return 1
    return 0
    



class StructuralAttack(Attack):
    """Class to wrap a number of attacks based on the static structure of a model."""

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable = too-many-arguments, too-many-locals
        self,
        risk_appetite_config:str="default",
        target_path: str = None,
    ) -> None:
        """Constructs an object to execute a structural attack.

        Parameters
        ----------
        report_name : str
            name of the pdf and json output reports
        target_path : str
            path to the saved trained target model and target data
        risk_appetite_config:str
            path to yaml file specifying TRE risk appetite
        """

        super().__init__()
        self.target:Target=None
        
        #make dummy acro object and use it to extract risk appetite
        myacro=ACRO(risk_appetite_config)
        self.THRESHOLD = myacro.config["safe_threshold"]
        self.DOF_THRESHOLD = myacro.config["safe_dof_threshold"] 
        del(myacro)
        
        #metrics
        self.DoF_risk = 0
        self.k_anonymity_risk=0
        self.class_disclosure_risk=0
        self.unnecessary_risk = 0
        logger.info(f'Thresholds for count {self.THRESHOLD} and DoF {self.DOF_THRESHOLD}')
        

    def __str__(self):
        return "Structural attack"

    def attack(self, target: Target) -> None:
        """Programmatic attack entry point.

        To be used when code has access to Target class and trained target model

        Parameters
        ----------
        target : attacks.target.Target
            target as a Target class object
        """
        self.target=target
        model_type="unknown"
        if target.model is not None:
            self.model_type = target.model
        else:
            errstr = ("cannot currently call StructuralAttack.attack() "
                      "unless the target contains a trained model"
                     )
            raise NotImplementedError(errstr)
        if isinstance(target.model,DecisionTreeClassifier):
            self.decision_tree_whitebox_attack()
        elif isinstance(target.model,RandomForestClassifier):
            self.random_forest_attack()
        elif isinstance(target.model,XGBClassifier):
            self.xgboost_attack()
        else:
            retstr= ("no current structural attacks "
                     f"for models of type {model_type}\n"
                    )
            logger.warning( retstr)
        
    def decision_tree_whitebox_attack(self)->None:
        """ Structural attacks on decision trees
        To be used when target model and training set are provided
        Tests for:
        - k-anonymity
        - class disclosure 
        - uneccessarily risky hyper-paramter combinations
        - residual degrees of freedom
        """

        #get tree structure
        dtree = self.target.model
        n_nodes = dtree.tree_.node_count
        left = dtree.tree_.children_left
        right = dtree.tree_.children_right
        is_leaf=np.zeros(n_nodes,dtype=int)
        for node_id in range( n_nodes):
            if left[node_id] == right[node_id]:
                is_leaf[node_id]=1
        n_leaves = is_leaf.sum()
        
        #degrees of freedom
        n_internal_nodes = n_nodes - n_leaves
        n_params= 2*n_internal_nodes #feature id and threshold
        n_params += n_leaves*( dtree.n_classes_ -1)#probability distribution
        self.residual_dof = self.target.x_train.shape[0]- n_params
        if self.residual_dof < self.DOF_THRESHOLD:
            self.DoF_risk = 1
        logger.debug(f'degrees of freedom for this decision tree is {self.residual_dof}, '
                   f'so DoF risk is {self.DoF_risk}')
  
        #find out which leaf nodes training data ends up in
        X = self.target.x_train
        y = self.target.y_train
        assert X.shape[0]==len(y),'data shape mismatch'
        destinations = dtree.apply(X)             
        
        # k-anonymity data
        uniqs_counts = np.unique(destinations, return_counts=True)
        leaves= uniqs_counts[0]
        n_leaves= len(leaves)
        #sanity check
        assert n_leaves == is_leaf.sum(),'mismatch counting leaves'
        #logger.info(f'There are {n_leaves} leaves in the tree')
        self.k_anonymity = np.min(uniqs_counts[1])
        if self.k_anonymity < self.THRESHOLD:
            self.k_anonymity_risk = 1
        logger.debug(f'minimum k-anonymity for this tree is {self.k_anonymity} '
                   f'so k-anonymity risk is {self.k_anonymity_risk}')
        
        #class disclosure
        leaf_membership = {}
        labels = np.unique(y,return_counts=False)
        n_classes = dtree.n_classes_
        #sanity checks
        assert n_classes == len(labels),'class count mismatch'
        assert (dtree.classes_ == labels).all(),'class label mismatch'    
        for leafidx in leaves:
            #initialise and zero dict to hold membership counts
            leaf_membership[leafidx]={}
            counts = {}
            for label in labels:
                counts[label]= 0
            for idx,val in enumerate(y):
                if destinations[idx] == leafidx:
                    counts[val] += 1
            leaf_membership[leafidx]=counts

        self.failing_regions = 0
        for key,val in leaf_membership.items():
            #logger.info(f'Leaf {key}, membership {val}')
            risky=False
            for label,count in val.items():
                #logger.info(f'key {key} label {label} count {count}')
                if count >0 and count <self.THRESHOLD:
                    risky=True
                    break
            if risky:
                self.failing_regions +=1
                #logger.debug(f'risky region: {key}:{val}')
        if self.failing_regions:
            self.class_disclosure_risk=1
        logger.debug(f' for this tree there are {self.failing_regions} problematic regions'
                    f' so class disclosure risk = {self.class_disclosure_risk}')
 
        # unnecessary risk arising from poor hyper-parameter combination.
        self.unnecessary_risk = unnecessary_risk(self.target.model,'dt')
        logger.debug(f'It is {self.unnecessary_risk==1} '
                     'that this model represents an unnecessary risk.'
                   )
    
    

#     def attack_from_prediction_files(self):
#         """Start an attack from saved prediction files.

#         To be used when only saved predictions are available.

#         Filenames for the saved prediction files to be specified in the arguments provided
#         in the constructor
#         """
#         train_preds = np.loadtxt(self.training_preds_filename, delimiter=",")
#         test_preds = np.loadtxt(self.test_preds_filename, delimiter=",")
#         self.attack_from_preds(train_preds, test_preds)

#     def attack_from_preds(  # pylint: disable=too-many-locals
#         self,
#         train_preds: np.ndarray,
#         test_preds: np.ndarray,
#         train_correct: np.ndarray = None,
#         test_correct: np.ndarray = None,
#     ) -> None:
#         """
#         Runs the attack based upon the predictions in train_preds and test_preds, and the params
#         stored in self.args.
#         This means that it is only possible to run blackbox attacks

#         Parameters
#         ----------
#         train_preds : np.ndarray
#             Array of train predictions. One row per example, one column per class (i.e. 2)
#         test_preds : np.ndarray
#             Array of test predictions. One row per example, one column per class (i.e. 2)
#         """
#         logger = logging.getLogger("attack-from-preds")
#         logger.info("Running main attack repetitions")
#         attack_metric_dict = self.run_attack_reps(
#             train_preds,
#             test_preds,
#             train_correct=train_correct,
#             test_correct=test_correct,
#         )
#         self.attack_metrics = attack_metric_dict["mia_metrics"]
#         self.attack_metric_failfast_summary = attack_metric_dict[
#             "failfast_metric_summary"
#         ]

#         self.dummy_attack_metrics = []
#         self.dummy_attack_metric_failfast_summary = []
#         if self.n_dummy_reps > 0:
#             logger.info("Running dummy attack reps")
#             n_train_rows = len(train_preds)
#             n_test_rows = len(test_preds)
#             for _ in range(self.n_dummy_reps):
#                 d_train_preds, d_test_preds = self.generate_arrays(
#                     n_train_rows,
#                     n_test_rows,
#                     self.train_beta,
#                     self.test_beta,
#                 )
#                 temp_attack_metric_dict = self.run_attack_reps(
#                     d_train_preds, d_test_preds
#                 )
#                 temp_metrics = temp_attack_metric_dict["mia_metrics"]
#                 temp_metric_failfast_summary = temp_attack_metric_dict[
#                     "failfast_metric_summary"
#                 ]

#                 self.dummy_attack_metrics.append(temp_metrics)
#                 self.dummy_attack_metric_failfast_summary.append(
#                     temp_metric_failfast_summary
#                 )

#         logger.info("Finished running attacks")

#     def _prepare_attack_data(
#         self,
#         train_preds: np.ndarray,
#         test_preds: np.ndarray,
#         train_correct: np.ndarray = None,
#         test_correct: np.ndarray = None,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """Prepare training data and labels for attack model
#         Combines the train and test preds into a single numpy array (optionally) sorting each
#         row to have the highest probabilities in the first column. Constructs a label array that
#         has ones corresponding to training rows and zeros to testing rows.
#         """
#         logger = logging.getLogger("prep-attack-data")
#         if self.sort_probs:
#             logger.info("Sorting probabilities to leave highest value in first column")
#             train_preds = -np.sort(-train_preds, axis=1)
#             test_preds = -np.sort(-test_preds, axis=1)

#         logger.info("Creating MIA data")

#         if self.include_model_correct_feature and train_correct is not None:
#             train_preds = np.hstack((train_preds, train_correct[:, None]))
#             test_preds = np.hstack((test_preds, test_correct[:, None]))

#         mi_x = np.vstack((train_preds, test_preds))
#         mi_y = np.hstack((np.ones(len(train_preds)), np.zeros(len(test_preds))))

#         return (mi_x, mi_y)

#     def run_attack_reps(  # pylint: disable = too-many-locals
#         self,
#         train_preds: np.ndarray,
#         test_preds: np.ndarray,
#         train_correct: np.ndarray = None,
#         test_correct: np.ndarray = None,
#     ) -> dict:
#         """
#         Run actual attack reps from train and test predictions.

#         Parameters
#         ----------
#         train_preds : np.ndarray
#             predictions from the model on training (in-sample) data
#         test_preds : np.ndarray
#             predictions from the model on testing (out-of-sample) data

#         Returns
#         -------
#         mia_metrics_dict : dict
#             a dictionary with two items including mia_metrics
#             (a list of metric across repetitions) and failfast_metric_summary object
#             (an object of FailFast class) to maintain summary of
#             fail/success of attacks for a given metric of failfast option
#         """
#         self.n_rows_in = len(train_preds)
#         self.n_rows_out = len(test_preds)
#         logger = logging.getLogger("attack-reps")
#         mi_x, mi_y = self._prepare_attack_data(
#             train_preds, test_preds, train_correct, test_correct
#         )

#         mia_metrics = []

#         failfast_metric_summary = FailFast(self)

#         for rep in range(self.n_reps):
#             logger.info("Rep %d of %d", rep + 1, self.n_reps)
#             mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(
#                 mi_x, mi_y, test_size=self.test_prop, stratify=mi_y
#             )
#             attack_classifier = self.mia_attack_model(**self.mia_attack_model_hyp)
#             attack_classifier.fit(mi_train_x, mi_train_y)
#             y_pred_proba, y_test = metrics.get_probabilities(
#                 attack_classifier, mi_test_x, mi_test_y, permute_rows=True
#             )

#             mia_metrics.append(metrics.get_metrics(y_pred_proba, y_test))

#             if self.include_model_correct_feature and train_correct is not None:
#                 # Compute the Yeom TPR and FPR
#                 yeom_preds = mi_test_x[:, -1]
#                 tn, fp, fn, tp = confusion_matrix(mi_test_y, yeom_preds).ravel()
#                 mia_metrics[-1]["yeom_tpr"] = tp / (tp + fn)
#                 mia_metrics[-1]["yeom_fpr"] = fp / (fp + tn)
#                 mia_metrics[-1]["yeom_advantage"] = (
#                     mia_metrics[-1]["yeom_tpr"] - mia_metrics[-1]["yeom_fpr"]
#                 )

#             failfast_metric_summary.check_attack_success(mia_metrics[rep])

#             if (
#                 failfast_metric_summary.check_overall_attack_success(self)
#                 and self.attack_fail_fast
#             ):
#                 break

#         logger.info("Finished simulating attacks")

#         mia_metrics_dict = {}
#         mia_metrics_dict["mia_metrics"] = mia_metrics
#         mia_metrics_dict["failfast_metric_summary"] = failfast_metric_summary

#         return mia_metrics_dict

    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Summarise metrics from a metric list.

        Returns
        -------
        global_metrics : Dict
            Dictionary of summary metrics

        Arguments
        ---------
        attack_metrics: List
            list of attack metrics dictionaries
        """
        global_metrics = {}
        if attack_metrics is not None and len(attack_metrics) != 0:
            auc_p_vals = [
                metrics.auc_p_val(
                    m["AUC"], m["n_pos_test_examples"], m["n_neg_test_examples"]
                )[0]
                for m in attack_metrics
            ]

            m = attack_metrics[0]
            _, auc_std = metrics.auc_p_val(
                0.5, m["n_pos_test_examples"], m["n_neg_test_examples"]
            )

            global_metrics[
                "null_auc_3sd_range"
            ] = f"{0.5 - 3*auc_std:.4f} -> {0.5 + 3*auc_std:.4f}"
            global_metrics["n_sig_auc_p_vals"] = self._get_n_significant(
                auc_p_vals, self.p_thresh
            )
            global_metrics["n_sig_auc_p_vals_corrected"] = self._get_n_significant(
                auc_p_vals, self.p_thresh, bh_fdr_correction=True
            )

            pdif_vals = [np.exp(-m["PDIF01"]) for m in attack_metrics]
            global_metrics["n_sig_pdif_vals"] = self._get_n_significant(
                pdif_vals, self.p_thresh
            )
            global_metrics["n_sig_pdif_vals_corrected"] = self._get_n_significant(
                pdif_vals, self.p_thresh, bh_fdr_correction=True
            )

        return global_metrics


    def _construct_metadata(self):
        """Constructs the metadata object, after attacks."""
        self.metadata = {}
        # Store all args
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"] = self.get_params()

        self.metadata["attack"] = str(self)

        # Global metrics
        self.metadata["global_metrics"] = self._get_global_metrics(self.attack_metrics)
        self.metadata["baseline_global_metrics"] = self._get_global_metrics(
            self._unpack_dummy_attack_metrics_experiments_instances()
        )


    def _get_attack_metrics_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}

        for rep, _ in enumerate(self.attack_metrics):
            attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]

        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        attack_metrics_experiment[
            "attack_metric_failfast_summary"
        ] = self.attack_metric_failfast_summary.get_attack_summary()

        return attack_metrics_experiment

    def _get_dummy_attack_metrics_experiments_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        dummy_attack_metrics_experiments = {}

        for exp_rep, _ in enumerate(self.dummy_attack_metrics):
            temp_dummy_attack_metrics = self.dummy_attack_metrics[exp_rep]
            dummy_attack_metric_instances = {}
            for rep, _ in enumerate(temp_dummy_attack_metrics):
                dummy_attack_metric_instances[
                    "instance_" + str(rep)
                ] = temp_dummy_attack_metrics[rep]
            temp = {}
            temp["attack_instance_logger"] = dummy_attack_metric_instances
            temp[
                "attack_metric_failfast_summary"
            ] = self.dummy_attack_metric_failfast_summary[exp_rep].get_attack_summary()
            dummy_attack_metrics_experiments[
                "dummy_attack_metrics_experiment_" + str(exp_rep)
            ] = temp

        return dummy_attack_metrics_experiments

    def make_report(self) -> dict:
        """Creates output dictionary structure and generates
        pdf and json outputs if filenames are given.
        """
        output = {}
        output["log_id"] = str(uuid.uuid4())
        output["log_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self._construct_metadata()
        output["metadata"] = self.metadata

        output["attack_experiment_logger"] = self._get_attack_metrics_instances()
        output[
            "dummy_attack_experiments_logger"
        ] = self._get_dummy_attack_metrics_experiments_instances()

        report_dest = os.path.join(self.output_dir, self.report_name)
        json_attack_formatter = GenerateJSONModule(report_dest + ".json")
        json_report = report.create_json_report(output)
        json_attack_formatter.add_attack_output(json_report, "WorstCaseAttack")

        pdf_report = report.create_mia_report(output)
        report.add_output_to_pdf(report_dest, pdf_report, "WorstCaseAttack")
        return output


def _make_dummy_data(args):
    """Initialise class and run dummy data creation."""
    args.__dict__["training_preds_filename"] = "train_preds.csv"
    args.__dict__["test_preds_filename"] = "test_preds.csv"
    attack_obj = WorstCaseAttack(
        train_beta=args.train_beta,
        test_beta=args.test_beta,
        n_rows_in=args.n_rows_in,
        n_rows_out=args.n_rows_out,
        training_preds_filename=args.training_preds_filename,
        test_preds_filename=args.test_preds_filename,
    )
    attack_obj.make_dummy_data()


def _run_attack(args):
    """Initialise class and run attack from prediction files."""
    attack_obj = WorstCaseAttack(
        n_reps=args.n_reps,
        p_thresh=args.p_thresh,
        n_dummy_reps=args.n_dummy_reps,
        train_beta=args.train_beta,
        test_beta=args.test_beta,
        test_prop=args.test_prop,
        training_preds_filename=args.training_preds_filename,
        test_preds_filename=args.test_preds_filename,
        output_dir=args.output_dir,
        report_name=args.report_name,
        sort_probs=args.sort_probs,
        attack_metric_success_name=args.attack_metric_success_name,
        attack_metric_success_thresh=args.attack_metric_success_thresh,
        attack_metric_success_comp_type=args.attack_metric_success_comp_type,
        attack_metric_success_count_thresh=args.attack_metric_success_count_thresh,
        attack_fail_fast=args.attack_fail_fast,
    )
    print(attack_obj.training_preds_filename)
    attack_obj.attack_from_prediction_files()
    _ = attack_obj.make_report()


def _run_attack_from_configfile(args):
    """Initialise class and run attack from prediction files using config file."""
    attack_obj = WorstCaseAttack(
        attack_config_json_file_name=str(args.attack_config_json_file_name),
        target_path=str(args.target_path),
    )
    target = Target()
    target.load(attack_obj.target_path)
    attack_obj.attack(target)
    _ = attack_obj.make_report()


def main():
    """Main method to parse arguments and invoke relevant method."""
    logger = logging.getLogger("main")
    parser = argparse.ArgumentParser(
        description=("Perform a worst case attack from saved model predictions")
    )

    subparsers = parser.add_subparsers()
    dummy_parser = subparsers.add_parser("make-dummy-data")
    dummy_parser.add_argument(
        "--num-rows-in",
        action="store",
        dest="n_rows_in",
        type=int,
        required=False,
        default=1000,
        help=("How many rows to generate in the in-sample file. Default = %(default)d"),
    )

    dummy_parser.add_argument(
        "--num-rows-out",
        action="store",
        dest="n_rows_out",
        type=int,
        required=False,
        default=1000,
        help=(
            "How many rows to generate in the out-of-sample file. Default = %(default)d"
        ),
    )

    dummy_parser.add_argument(
        "--train-beta",
        action="store",
        type=float,
        required=False,
        default=5,
        dest="train_beta",
        help=(
            """Value of b parameter for beta distribution used to sample the in-sample
            probabilities. High values will give more extreme probabilities. Set this
            value higher than --test-beta to see successful attacks. Default = %(default)f"""
        ),
    )

    dummy_parser.add_argument(
        "--test-beta",
        action="store",
        type=float,
        required=False,
        default=2,
        dest="test_beta",
        help=(
            "Value of b parameter for beta distribution used to sample the out-of-sample "
            "probabilities. High values will give more extreme probabilities. Set this value "
            "lower than --train-beta to see successful attacks. Default = %(default)f"
        ),
    )

    dummy_parser.set_defaults(func=_make_dummy_data)

    attack_parser = subparsers.add_parser("run-attack")
    attack_parser.add_argument(
        "-i",
        "--training-preds-filename",
        action="store",
        dest="training_preds_filename",
        required=False,
        type=str,
        default="train_preds.csv",
        help=(
            "csv file containing the predictive probabilities (one column per class) for the "
            "training data (one row per training example). Default = %(default)s"
        ),
    )

    attack_parser.add_argument(
        "-o",
        "--test-preds-filename",
        action="store",
        dest="test_preds_filename",
        required=False,
        type=str,
        default="test_preds.csv",
        help=(
            "csv file containing the predictive probabilities (one column per class) for the "
            "non-training data (one row per training example). Default = %(default)s"
        ),
    )

    attack_parser.add_argument(
        "-r",
        "--n-reps",
        type=int,
        required=False,
        default=5,
        action="store",
        dest="n_reps",
        help=(
            "Number of repetitions (splitting data into attack model training and testing "
            "partitions to perform. Default = %(default)d"
        ),
    )

    attack_parser.add_argument(
        "-t",
        "--test-prop",
        type=float,
        required=False,
        default=0.3,
        action="store",
        dest="test_prop",
        help=(
            "Proportion of examples to be used for testing when fiting the attack model. "
            "Default = %(default)f"
        ),
    )

    attack_parser.add_argument(
        "--output-dir",
        type=str,
        action="store",
        dest="output_dir",
        default="output_worstcase",
        required=False,
        help=("Directory name where output files are stored. Default = %(default)s."),
    )

    attack_parser.add_argument(
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        default="report_worstcase",
        required=False,
        help=(
            """Filename for the pdf and json report outputs. Default = %(default)s.
            Code will append .pdf and .json"""
        ),
    )

    attack_parser.add_argument(
        "--n-dummy-reps",
        type=int,
        action="store",
        dest="n_dummy_reps",
        default=1,
        required=False,
        help=(
            "Number of dummy datasets to sample. Each will be assessed with --n-reps train and "
            "test splits. Set to 0 to do no baseline calculations. Default = %(default)d"
        ),
    )

    attack_parser.add_argument(
        "--p-thresh",
        action="store",
        type=float,
        default=P_THRESH,
        required=False,
        dest="p_thresh",
        help=("P-value threshold for significance testing. Default = %(default)f"),
    )

    attack_parser.add_argument(
        "--train-beta",
        action="store",
        type=float,
        required=False,
        default=5,
        dest="train_beta",
        help=(
            "Value of b parameter for beta distribution used to sample the in-sample probabilities."
            "High values will give more extreme probabilities. Set this value higher than "
            "--test-beta to see successful attacks. Default = %(default)f"
        ),
    )

    attack_parser.add_argument(
        "--test-beta",
        action="store",
        type=float,
        required=False,
        default=2,
        dest="test_beta",
        help=(
            "Value of b parameter for beta distribution used to sample the out-of-sample "
            "probabilities. High values will give more extreme probabilities. Set this value "
            "lower than --train-beta to see successful attacks. Default = %(default)f"
        ),
    )

    # --include-correct feature not supported as not currently possible from the command line
    # as we cannot compute the correctness of predictions.

    attack_parser.add_argument(
        "--sort-probs",
        action="store",
        type=bool,
        default=True,
        required=False,
        dest="sort_probs",
        help=(
            "Whether or not to sort the output probabilities (per row) before "
            "using them to train the attack model. Default = %(default)f"
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-name",
        action="store",
        type=str,
        default="P_HIGHER_AUC",
        required=False,
        dest="attack_metric_success_name",
        help=(
            """for computing attack success/failure based on
            --attack-metric-success-thresh option. Default = %(default)s"""
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-thresh",
        action="store",
        type=float,
        default=0.05,
        required=False,
        dest="attack_metric_success_thresh",
        help=(
            """for defining threshold value to measure attack success
            for the metric defined by argument --fail-metric-name option. Default = %(default)f"""
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-comp-type",
        action="store",
        type=str,
        default="lte",
        required=False,
        dest="attack_metric_success_comp_type",
        help=(
            """for computing attack success/failure based on
            --attack-metric-success-thresh option. Default = %(default)s"""
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-count-thresh",
        action="store",
        type=int,
        default=2,
        required=False,
        dest="attack_metric_success_count_thresh",
        help=(
            """for setting counter limit to stop further repetitions given the attack is
             successful and the --attack-fail-fast is true. Default = %(default)d"""
        ),
    )

    attack_parser.add_argument(
        "--attack-fail-fast",
        action="store_true",
        required=False,
        dest="attack_fail_fast",
        help=(
            """to stop further repetitions when the given metric has fulfilled
            a criteria for a specified number of times (--attack-metric-success-count-thresh)
            and this has a true status. Default = %(default)s"""
        ),
    )

    attack_parser.set_defaults(func=_run_attack)

    attack_parser_config = subparsers.add_parser("run-attack-from-configfile")
    attack_parser_config.add_argument(
        "-j",
        "--attack-config-json-file-name",
        action="store",
        required=True,
        dest="attack_config_json_file_name",
        type=str,
        default="config_worstcase_cmd.json",
        help=(
            "Name of the .json file containing details for the run. Default = %(default)s"
        ),
    )

    attack_parser_config.add_argument(
        "-t",
        "--attack-target-folder-path",
        action="store",
        required=True,
        dest="target_path",
        type=str,
        default="worstcase_target",
        help=(
            """Name of the target directory to load the trained target model and the target data.
            Default = %(default)s"""
        ),
    )

    attack_parser_config.set_defaults(func=_run_attack_from_configfile)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError as e:  # pragma:no cover
        logger.error("Invalid command. Try --help to get more details")
        logger.error(e)


if __name__ == "__main__":  # pragma:no cover
    main()
