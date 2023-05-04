"""
This module presents example attacks a TRE checker may perform.
"""

import sys
import os
import logging
import pickle
import numpy as np

from aisdc.attacks.dataset import Data
from aisdc.attacks.worst_case_attack import WorstCaseAttack, WorstCaseAttackArgs
from aisdc.generate_report import process_json

DIR = "training_artefacts/"
print("Creating directory for training artefacts")

if not os.path.exists(DIR):
    os.makedirs(DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

# Suppress messages from AI-SDC -- comment out these lines to see all the aisdc logging statements
logging.getLogger("attack-reps").setLevel(logging.WARNING)
logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

try:
    FILENAME = f"{DIR}/disclosive_random_forest.sav"
    print(f"Reading disclosive random forest from {FILENAME}")
    with open(FILENAME, "rb") as fp:
        target_model = pickle.load(fp)
except FileNotFoundError:
    print("Target model not found - please run sacro_nursery_researcher.py")
    sys.exit(0)

print(f"Reading training/testing data from ./{DIR}")
trainX = np.loadtxt(DIR + "trainX.txt")
trainy = np.loadtxt(DIR + "trainy.txt")
testX = np.loadtxt(DIR + "testX.txt")
testy = np.loadtxt(DIR + "testy.txt")

sdc_data = Data()
# Wrap the training and test data into the Data object
sdc_data.add_processed_data(trainX, trainy, testX, testy)
# Create attack args.
args = WorstCaseAttackArgs(n_dummy_reps=0, report_name=DIR + "/disclosive_model_output")
# Run the attack
wca = WorstCaseAttack(args)
wca.attack(sdc_data, target_model)

json_out = wca.make_report()

DISCLOSIVE_FILENAME = "disclosive_model_summary.txt"
process_json(DIR + "/disclosive_model_output.json", DISCLOSIVE_FILENAME)

print()

FILENAME = f"{DIR}/safe_random_forest.sav"
print(f"Reading safe random forest from {FILENAME}")
with open(FILENAME, "rb") as fp:
    target_model = pickle.load(fp)

sdc_data = Data()
sdc_data.add_processed_data(trainX, trainy, testX, testy)
args = WorstCaseAttackArgs(n_dummy_reps=0, report_name=DIR + "/safe_model_output")
wca = WorstCaseAttack(args)

# Suppress messages from AI-SDC
logging.getLogger("attack-reps").setLevel(logging.WARNING)
logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

wca.attack(sdc_data, target_model)

json_out = wca.make_report()

SAFE_FILENAME = "safe_model_summary.txt"
process_json(DIR + "/safe_model_output.json", SAFE_FILENAME)

print()
print(
    "=============================="
    f"\nReports written to {SAFE_FILENAME} + and {DISCLOSIVE_FILENAME}"
    "\nDESCRIPTION OF FILES"
    "\nPlease note: this is a draft of a possible output that could be produced "
    "- feedback is appreciated!"
    "\nfinal_score (or summary risk level): a score from 0-5"
    "\n     a score of 5 means the model is highly disclosive"
    "\n     a score of 1 (or an empty file) means the model is not found to be disclosive"
    "\nscore_breakdown: for each of the tests that were run and indicated a disclosive "
    "model, a score of the impact on risk is provided (1-5, 5 is most disclosive)."
    "\n     This list is used to calculate the final_score above"
    "\nscore_descriptions: for each tests that were run and indicated a disclosive model,"
    "a description of the test is provided"
    "\n     This information can be used by the researcher to improve their models"
)
