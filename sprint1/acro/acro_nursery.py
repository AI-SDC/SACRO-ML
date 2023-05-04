"""
ACRO Tests
Copyright : Maha Albashir, Richard Preen, Jim Smith 2023
"""

# import libraries
import os

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from acro import ACRO, add_constant

# ### Instantiate ACRO by making an acro object
print(
    "\n Creating an acro object().\n"
    "The TRE's risk appetite is read from default.yml\n"
    "and shown to the researcher and output checker"
)
acro = ACRO()


"""
Load test data
The dataset used in this notebook is the nursery dataset from OpenML.
 - The dataset can be read directly from OpenML using the code commented in the next cell.
 - In this version, it can be read directly from the local machine
   if it has been downloaded.
- The code below reads the data from a folder called "data"
  which we assume is at the same level as the folder where you are working.
 - The path might need to be changed if the data has been downloaded and stored elsewhere.
 - for example use:
    path = os.path.join("data", "nursery.arff")
   if the data is in a sub-folder of your work folder
"""


"""
commented out version to load from web
# from sklearn.datasets import fetch_openml
# data = fetch_openml(data_id=26, as_frame=True)
# df = data.data
# df["recommend"] = data.target
"""

# Version to load data from local directory
path = os.path.join("../data", "dataset_26_nursery.arff")
data = loadarff(path)
df = pd.DataFrame(data[0])
df = df.select_dtypes([object])
df = df.stack().str.decode("utf-8").unstack()
df.rename(columns={"class": "recommend"}, inplace=True)

print("\n Data loaded, these are the first five rows")
print(df.head())


""" Convert 'more than 3' children to random between 4 and 10
Change the children column from categorical to numeric
in order to be able to test some of the ACRO functions that require a numeric feature
"""
print("\nChanging number of children to integer type")
df["children"].replace(to_replace={"more": "4"}, inplace=True)
df["children"] = pd.to_numeric(df["children"])

df["children"] = df.apply(
    lambda row: row["children"]
    if row["children"] in (1, 2, 3)
    else np.random.randint(4, 10),
    axis=1,
)


"""
Examples of producing tabular output
 We rely on the industry-standard package **pandas** for tabulating data.
 In the next few examples we show:
 - first, how a researcher would normally make a call in pandas,
   saving the results in a variable that they can view on screen (or save to file?)
 - then how the call is identical in SACRO, except that:
   - "pd" is replaced by "acro"
   - the researcher immediately sees  TRE output checking recommendations.
"""
print(
    "\nThe first set of examples show acro wrappers around "
    " standard tabulation routines from the pandas package."
)

"""
Pandas crosstab
 This is an example of crosstab using pandas.
 We first make the call, then the second line print the outputs to wscreen.
"""
print("\nCalling crosstab of recommendation by parents using pandas")
table = pd.crosstab(df.recommend, df.parents)
print(table)


"""
ACRO crosstab
 This is an example of crosstab using ACRO.
 The INFO lines show the researcher what will be reported to the output checkers.
 Then the (suppressed as necessary)  table is shown via. the print command as before.
"""
print("\nNow the same crosstab call using the ACRO interface")
safe_table = acro.crosstab(df.recommend, df.parents)
print("\nand this is the researchers output")
print(safe_table)

"""
ACRO crosstab with aggregation function
Mean() in this case
Then how Max and Min are not allowed by the code
"""
print(
    "\nIllustration of crosstab using an aggregation function " "- mean in this case."
)
safe_table = acro.crosstab(df.recommend, df.parents, values=df.children, aggfunc="mean")
print("\nand this is the researchers output")
print(safe_table)

print(
    "\nThis is what happens if you try to get max values for a cell."
    "\nSo that this script runs on one go, we've caught the exception "
    "thrown by ACRO."
)
try:
    safe_table = acro.crosstab(
        df.recommend, df.parents, values=df.children, aggfunc="max"
    )
except ValueError as e:
    print("ValueError:")
    print(e)

"""
ACRO pivot_table
This is an example of pivot table using ACRO.
 - Some researchers may prefer this to using crosstab.
 - Again the call syntax is identical to the pandas "pd.pivot_table"
 - in this case the output is non-disclosive
"""
print("\nIllustration of using the acro version of pandas pivot table")
table = acro.pivot_table(
    df, index=["parents"], values=["children"], aggfunc=["mean", "std"]
)
print("\nand this is the researchers output")
print(table)


"""
Regression examples using ACRO
Again there is an industry-standard package in python, this time called **statsmodels**.
 - The examples below illustrate the use of the ACRO wrapper standard statsmodel functions
 - Note that statsmodels can be called using an 'R-like' format
   (using an 'r' suffix on the command names)
 - most statsmodels functions return a "results object",
   which has a "summary" function that produces printable/saveable outputs
"""
print(
    "\nThe next set of examples illustrate acro wrappers "
    "around functions from the statsmodels package"
)

"""
Start by manipulating the nursery data to get two numeric variables
 - The 'recommend' column is converted to an integer scale
"""

df["recommend"].replace(
    to_replace={
        "not_recom": "0",
        "recommend": "1",
        "very_recom": "2",
        "priority": "3",
        "spec_prior": "4",
    },
    inplace=True,
)
df["recommend"] = pd.to_numeric(df["recommend"])

new_df = df[["recommend", "children"]]
new_df = new_df.dropna()


"""
ACRO OLS
 This is an example of ordinary least square regression using ACRO.
 - Above recommend column was converted form categorical to numeric.
 - Now we perform a the linear regression between recommend and children.
 - This version includes a constant (intercept)
 - This is just to show how the regression is done using ACRO.
 - **No correlation is expected to be seen by using these variables**
"""

y = new_df["recommend"]
x = new_df["children"]
x = add_constant(x)
print("\nOrdinary Least Squares Regression")
results = acro.ols(y, x)
print("\nand this is the researchers output")
print(results.summary())


"""
ACRO OLSR
This is an example of ordinary least squares regression using the 'R-like' statsmodels api,
i.e. from a formula and dataframe using ACRO
"""
print("\nAnd same, but  passing a formula instead of two arrays")
results = acro.olsr(formula="recommend ~ children", data=new_df)
print("\nand this is the researchers output")
print(results.summary())


"""
ACRO Probit
 This is an example of probit regression using ACRO
 We use a different combination of variables from the original dataset.
"""
new_df = df[["finance", "children"]]
new_df = new_df.dropna()

y = new_df["finance"].astype("category").cat.codes  # numeric
y.name = "finance"
x = new_df["children"]
x = add_constant(x)
print("\n Example of a probit regression")
results = acro.probit(y, x)
print("\nand this is the researchers output")
print(results.summary())


"""
ACRO Logit
This is an example of logistic regression using ACRO using the statmodels function
"""
print("\n Example of a logit regression")
results = acro.logit(y, x)
print("\nand this is the researchers output")
print(results.summary())


"""
ACRO functionality to let users manage their outputs

1: List current ACRO outputs
 This is an example of using the print_output function to list all the outputs created so far
"""
print("\nNow illustrating how users can manage their outputs")
print(
    "\nStart by listing the outputs in the acro memory."
    "For each output the key line is the one starting 'Summary'"
)
acro.print_outputs()


"""
2: Remove some ACRO outputs before finalising
 This is an example of deleting some of the ACRO outputs.
 The name of the output  to be removed should be passed to the function remove_output.
 - Currently, all outputs names contain timestamp;
   that is the time when the output was created.
 - The output name can be taken from the outputs listed by the print_outputs function,
 - or by listing the results and choosing the specific output that needs to be removed
"""
print("\nNow removing two disclosive outputs")
output_1 = list(acro.results.keys())[1]
output_4 = list(acro.results.keys())[4]

acro.remove_output(output_1)
acro.remove_output(output_4)


"""
3: Rename ACRO outputs before finalising
 This is an example of renaming the outputs to provide a more descriptive name.
 The timestamp associated with the output name will not get overwritten
"""
print("\nUsers can rename output files to something more informative")
acro.rename_output(list(acro.results.keys())[2], "pivot_table")


"""
4: Add a comment to output
 This is an example to add a comment to outputs.
 It can be used to provide a description
 or to pass additional information to the output checkers.
"""
print("\nUsers can add comments which the output checkers will see.")
acro.add_comments(list(acro.results.keys())[0], "Please let me have this table!")
acro.add_comments(list(acro.results.keys())[0], "6 cells were suppressed in this table")


"""
5: Add an unsupported output to the list of outputs
 This is an example to add an unsupported outputs (such as images) to the list of outputs
"""
print("\nUsers can add files produced by an analysis aCRO doesn't cover")
acro.custom_output(
    "XandY.jfif", "This output is an image showing the relationship between X and Y"
)


"""
6 (the big one) Finalise ACRO
 This is an example of the function _finalise()_
 which the users must call at the end of each session.
 - It takes each output and saves it to a CSV file.
 - It also saves the SDC analysis for each output to a json file or Excel file
   (depending on the extension of the name of the file provided as an input to the function)
"""
print(
    "\nUsers MUST call finalise to send their outputs to the checkers"
    " If they don't, the SDC analysis, and their outputs, are lost."
)
output = acro.finalise("test_results.json")
