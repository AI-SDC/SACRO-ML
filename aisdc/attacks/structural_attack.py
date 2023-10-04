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
def get_rf_k_anonymity(model: RandomForestClassifier, x: np.ndarray) -> int:
    """Calculates the k-anonymity of a random forest model
    as the minimum of the anonymity for each record.
    That is defined as the size of the set of records which
    appear in the same leaf as the record in every tree.
    """

    # dataset must be 2-D
    assert len(x.shape) == 2

    num_records = x.shape[0]
    num_trees = model.n_estimators
    k_anon_val = np.zeros(num_records, dtype=int)

    # ending leaf node by record(row) and tree (column)
    all_leaves = np.zeros((num_records, num_trees), dtype=int)
    for this_tree in range(num_trees):
        this_leaves = model.estimators_[this_tree].apply(x)
        for record in range(num_records):
            all_leaves[record][this_tree] = this_leaves[record]

    for record in range(num_records):
        # start by assuming everything co-occurs
        appears_together = list(range(0, num_records))
        # iterate through trees
        for this_tree in range(num_trees):
            this_leaf = all_leaves[record][this_tree]

            together = copy.copy(appears_together)
            # removing records which go to other leaves
            for other_record in together:
                if all_leaves[other_record][this_tree] != this_leaf:
                    appears_together.remove(other_record)

        k_anon_val[record] = len(appears_together)
    return k_anon_val.min()


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
        if (max_depth > 3.5 and n_estimators > 35 and min_samples_split <= 15 and
            max_features == None and model.bootstrap):
            return 1
        if (max_depth > 7.5 and 15 < n_estimators <= 35 and 
            min_samples_leaf <= 15 and not model.bootstrap):
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
        output_dir = "outputs_structural",
        report_name="report_structural"
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
        self.target_path = target_path
        
        #make dummy acro object and use it to extract risk appetite
        myacro=ACRO(risk_appetite_config)
        self.risk_appetite_config= risk_appetite_config
        self.THRESHOLD = myacro.config["safe_threshold"]
        self.DOF_THRESHOLD = myacro.config["safe_dof_threshold"] 
        del(myacro)
        
        #metrics
        self.attack_metrics= ['DoF_risk','k_anonymity_risk','class_disclosure_risk','unnecessry_risk']
        self.DoF_risk = 0
        self.k_anonymity_risk=0
        self.class_disclosure_risk=0
        self.unnecessary_risk = 0
        logger.info(f'Thresholds for count {self.THRESHOLD} and DoF {self.DOF_THRESHOLD}')
        
        #paths for reporting
        self.output_dir= output_dir 
        self.report_name = report_name

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
        # if isinstance(target.model,DecisionTreeClassifier):
        #     self.decision_tree_whitebox_attack()
        # elif isinstance(target.model,RandomForestClassifier):
        #     self.random_forest_attack()
        # elif isinstance(target.model,XGBClassifier):
        #     self.xgboost_attack()
        # else:
        #     retstr= ("no current structural attacks "
        #              f"for models of type {model_type}\n"
        #             )
        #     logger.warning( retstr)
        # get proba values for training data    
        x = self.target.x_train
        y = self.target.y_train
        if len(y.shape)==1:
            n_classes = len( np.unique(y))
        else:
            nclasses=y.shape[1]
        n_rows=x.shape[0]
        assert x.shape[0]==len(y),'length mismatch between trainx and trainy'
        self.yprobs = self.target.model.predict_proba(x)
        
        #only equivalance classes and membership once as potentially slow
        if isinstance(target.model,DecisionTreeClassifier):
            equiv = self.dt_get_equivalence_classes()
            self.equiv_classes= equiv[0]
            self.equiv_counts =equiv[1]
            self.equiv_members = equiv[2]
        else:
            uniques = np.unique(self.yprobs,axis=0,return_counts=True)
            self.equiv_classes = uniques[0]
            self.equiv_counts = uniques[1]
            self.equiv_members = self.get_equivalence_membership()
 
            
    def dt_get_equivalence_classes(self)->tuple:
        """ to complete"""
        destinations = self.target.model.apply(self.target.x_train)
        ret_tuple= np.unique(destinations,return_counts=True)
        #print(f'leaves and counts:\n{ret_tuple}\n')
        leaves = ret_tuple[0]
        counts= ret_tuple[1]
        members = []
        for leaf in leaves:
            ingroup = np.asarray(destinations==leaf).nonzero()[0]
            print(f'ingroup {ingroup},count {len(ingroup)}')
            members.append(ingroup)
        return [leaves,counts,members]

    def get_equivalence_membership(self)->list:
        """ to complete"""
        members=[]
        for prob_vals in self.equiv_classes:
            ingroup = np.unique(np.asarray(self.yprobs==prob_vals).nonzero()[0])
            members.append(ingroup)
        return members
    
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
    
#     def random_forest_attack(self)->None:
#         """ runs structural attacks on random forests """
        
#         forest = self.target.model
#         x= self.target.x_train
        
#         # k-anonymity
#         kval = get_rf_k_anonymity(forest,x)
#         if kval < self.THRESHOLD:
#             self.k_anonymity_risk= 1
        
#         # class disclosure- step 1 get prediction probabilities
#         yprobs = forest.prediction_proba(x).sort()
#         assert isinstance(np.ndarray, yprobs)
#         assert yprobs.shape[0] == x.shape[0]
#         assert yprobs.shape[1] == forest._n_classes
        
        
#         #class disclosure step 2:next sort prediction_probs by first col
#         #then second etc preserving rows
#         # method from https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
#         sorted_probs = yprobs[np.lexsort(([yprobs[:, i] for i in range(yprobs.shape[1]-1, -1, -1)]))]
        
#         #class disclosure step 3:loop through all similarity groups
#         group_first = 0
#         group_last= 0
#         while group_last<sorted_probs.shape[0] -1:
#             possible_next=group_last+1
#             #get group of records with identical prob_a values
#             while sorted_probs[possible_next]==sorted_probs[group_first]
#                 group_last +=1
#                 possible_next +=1
            
        
        
#         # unnecessary risk arising from poor hyper-parameter combination.
#         self.unnecessary_risk = unnecessary_risk(forest,'rf')
#         logger.debug(f'It is {self.unnecessary_risk==1} '
#                      'that this model represents an unnecessary risk.'
#                    )
        
#     def xgboost_attack(self)->None:
#         """ runs structural attacks on xgboost forests """
        
        
#         #not srue class disclosure is meaningful
        
        
#         # unnecessary risk arising from poor hyper-parameter combination.
#         self.unnecessary_risk = unnecessary_risk(self.target.model,'xgboost')
#         logger.debug(f'It is {self.unnecessary_risk==1} '
#                      'that this model represents an unnecessary risk.'
#                    )
        
        



    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Summarise metrics from a metric list.

        Returns
        -------
        global_metrics : Dict
            Dictionary of summary metrics

        Arguments
        ---------
        attack_metrics: List
            list of attack metrics to be reported
        """
        global_metrics = {}
        if attack_metrics is not None and len(attack_metrics) != 0:
            global_metrics["DoF_risk"] =self.DoF_risk
            global_metrics["k_anonymity_risk"]= self.k_anonymity_risk
            global_metrics["class_disclosure_risk"]=self.class_disclosure_risk
            global_metrics["unnecessary_risk"]= self.unnecessary_risk 
            

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
################################################


    def _get_attack_metrics_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}

        #for rep, name in enumerate(self.attack_metrics):
        #    #attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]
        #    attack_metrics_instances["instance_" + str(name)] = self.__dict__.[name]

        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        attack_metrics_experiment["DoF_risk"] =self.DoF_risk
        attack_metrics_experiment["k_anonymity_risk"]= self.k_anonymity_risk
        attack_metrics_experiment["class_disclosure_risk"]=self.class_disclosure_risk
        attack_metrics_experiment["unnecessary_risk"]= self.unnecessary_risk 

        return attack_metrics_experiment



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
        # output[
        #     "dummy_attack_experiments_logger"
        # ] = self._get_dummy_attack_metrics_experiments_instances()

        report_dest = os.path.join(self.output_dir, self.report_name)
        json_attack_formatter = GenerateJSONModule(report_dest + ".json")
        json_report = report.create_json_report(output)
        json_attack_formatter.add_attack_output(json_report, "StructuralAttack")

        #pdf_report = report.create_mia_report(output)
        #report.add_output_to_pdf(report_dest, pdf_report, "StructuralAttack")
        return output




def _run_attack(args):
    """Initialise class and run attack from prediction files."""
    
    #        risk_appetite_config:str="default",
    #    target_path: str = None,
    attack_obj = StructuralAttack(
        risk_appetite_config:=args.risk_appetite_config,
        target_path=args.target_path,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(attack_obj.training_preds_filename)
    attack_obj.attack_from_prediction_files()
    _ = attack_obj.make_report()


def _run_attack_from_configfile(args):
    """Initialise class and run attack from prediction files using config file."""
    
    attack_obj = StructuralAttack(
        risk_appetite_config=str(args.risk_appetite_config),
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
        description=("Perform a structural  attack from saved model predictions")
    )

    subparsers = parser.add_subparsers()

    attack_parser = subparsers.add_parser("run-attack")

    attack_parser.add_argument(
        "--output-dir",
        type=str,
        action="store",
        dest="output_dir",
        default="output_structural",
        required=False,
        help=("Directory name where output files are stored. Default = %(default)s."),
    )

    attack_parser.add_argument(
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        default="report_structural",
        required=False,
        help=(
            """Filename for the pdf and json report outputs. Default = %(default)s.
            Code will append .pdf and .json"""
        ),
    )


    attack_parser.add_argument(
        "--risk-appetite-filename",
        action="store",
        type=str,
        default="default",
        required=False,
        dest="risk_appetite_config",
        help=(
            """provide the name of the dataset-specific risk appetite filename
            using --risk-appetite-filename Default = %(default)s"""
        ),
    )

    attack_parser.add_argument(
        "--target-path",
        action="store",
        type=float,
        default=None,
        required=False,
        dest="target_path",
        help=(
            """Provide the path to the stored target usinmg
             --target-path option. Default = %(default)f"""
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
        default="config_structural_cmd.json",
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
        default="structural_target",
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
