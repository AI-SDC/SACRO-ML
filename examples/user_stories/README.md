# User Stories

In this section there are code examples of how the SACRO-ML tools can be used by both a researchers in Trusted Research Environment (TRE) and a TRE output checkers. Each project is unique and therefore how SACRO-ML tools are applied may vary from case to case. The user guides have been split into 8 'user stories', each designed to fit a different use-case.

The following diagram is intended to identify the closest use-case match to projects:

![User Stories](user_stories_flow_chart.drawio.png)

## General description

The user stories are coding examples intended to maximise the chances of  successfully and smoothly egressing a Machine Learning (ML) model from the TRE. These guides are useful to create appropriate ML models and the metadata files necessary for output checking of the ML model prior to the egress. Saving time and effort, and ultimately optimising costs.

Each user story consists of at least 2 files:
  > - **user_story_[x]_researcher_template.[py/R]**  Example on how to generate a ML model for the TRE users/researchers.
  > - **user_story_[x]_tre.py**  Example on how to perform attacks and generate a report.

Extra examples on how to use [safemodels](https://github.com/AI-SDC/SACRO-ML/tree/main/examples/notebooks) and perform [attacks](https://github.com/AI-SDC/SACRO-ML/tree/main/examples) can be found following the corresponding links.

## Programming languages

Although, SACRO-ML tools is written in Python, some projects may use a different programming language to create their target model. However, where possible Python should be preferred as more extensive risk-disclosure testing has been performed.

While most of the stories are Python examples, `user_story_4` is written in R.

## Instructions

**For researchers or users**
1. Select the best use-story match to the project.
2. Familiarise yourself with the relevant user-story example, and discuss this with the TRE. Understanding how the process work for both sides will increase the changes of smooth project.
3. Pre-process data and generate the ML model as appropriate for the project inside the TRE. Remember to follow the relevant researcher  user story example code (**user_story_[x]_researcher.[py/R]**).
4. Make sure you generated all metadata, data and files required for output checking.
5. Fill out the `default_config.yaml` with the appropriate fields. An example of this file can be found [here](https://github.com/AI-SDC/SACRO-ML/blob/user_story_visibility/user_stories/default_config.yaml) with required experiment parameters.
6. Run the command `python generate_disclosure_risk_report.py`.
7. View all the required output files in the **release_files** folder, where the all the required files, data and metadata for egress are placed. A folder called **training_artefacts** is also created, this will include training and testing data and any detailed results of attacks.

*Alternative to steps 5 and 6*

5. Create a new configuration file using the same format in the [default_config.yaml](https://github.com/AI-SDC/SACRO-ML/blob/main/examples/user_stories/default_config.yaml) file with a different name.
6. Run the command `python generate_disclosure_risk_report.py --config <your_config_file_name>`.

**For TRE output checkers**
1. Select the best use-story match to the project. Preferably, this should have been agreed with the user/researcher beforehand.
2. Familiarise yourself with the relevant user-story example, both for researchers and TRE. Understanding how the process work both sides will increase the chances of a smoothly running project.
3. Once the researcher/user requests the ML model egress, perform attacks according to corresponding **user_story_[x]_tre.py** example.
4. Look at the reports produced and make a judgment for model egress.

## The user stories in detail

Unless otherwise specified, the stories are for Python machine learning models.

### User story 1: Ideal Case

The user is familiar with SACRO-ML tools. Also, the ML classifiers chosen for the project has been wrapped with the SafeModel class. This class, as the name indicates, ensures that the most leaky ML hyperparameters cannot be set, and therefore reducing the risk of data leakage from the generated model. Moreover, the user created the `Target` object provided by SACRO-ML tools. This ensures an easier process to generate the data and metadata files required for the model release.

The user can perform attacks to check the viability of their model for release and have the opportunity to make changes where necessary. Once the user is satisfied, and generated all the attacks, the TRE staff performs an output check and makes a decision. This way the user optimises the time for release.

### User story 2: SafeModel class and Target object employed

The user is familiar with SACRO-ML tools. Also, the ML classifiers chosen for the project has been wrapped with the SafeModel class. This class, as the name indicates, ensures that the most leaky ML hyperparameters cannot be set, and therefore reducing the risk of data leakage from the generated model.

In this example, the user does not use the `Target` object provided by SACRO-ML tools, and does not call the function function `request_release` which provides all the data and metadata files required for output check. This means that the output checker has to recreate the processed data (code provided by user). The user also needs to state which rows of the data were used for training and testing of the model. Once all of this is fulfilled, the output checker can run attacks, generate reports and make a decision on model release.

### User Story 3: User provides dataset object but does not use SafeModel

There exist a vast number of classifiers available and only for a few the SafeModle wrapper exists. Therefore, for some purposes will not be possible to use the SafeModel class.

However, the user has provided a copy of their training data alongside the model to be released. By using this package, the TRE can therefore check the hyperparameters of the model, as well as running attacks and generating reports which will help the TRE to make a decision regarding whether the model should be released.

### User Story 4: User does not use safeXClassifier, or provide dataset object
#### but does provide description of pre-processing, and provides output probabilities for the train and test set they have used

In this example, a researcher has a model (written in Python or R for example) which makes a prediction based on some data. The researcher has not provided a copy of their training data, but has provided a list of output probabilities for each class their model predicts, for each sample in their dataset, in a .csv file format.

The TRE, by using this package and this user story, can run some of the attacks available in this package. Doing so will generate a report, which will help the TRE to make a decision on whether the model should be released.

### User Story 5:  User creates differentially private algorithm (not via our code) and provides sufficient details to create data object.
##### Status: not yet implemented

In this example, a researcher has built a differentially private algorithm, but no details of training/testing data.

At time of writing (October 2023), we are not currently in a position to be able to automate epsilon value claims.

Additionally, some packages include disclosive values as metadata embedded in their models, which would need to be extracted and removed prior to release.

We are therefore not able to recommend release of these models at time of writing, although this work is still ongoing.

### User Story 6: Worst Case
##### Status: not yet implemented

In this example, a researcher has built a model which has not yet been tested by the aisdc package, and have not provided details of their training or testing data.

At time of writing (October 2023), experiments are still being done to determine what we can tell in terms of class disclosure.

Therefore, this user story is still under experimentation/implementation.

### User Story 7: User provides safemodel with no data

In this example, a user builds a model using the SafeModel class, and wraps their data in a Target object. However, the researcher forgets to call request_release() or Target.save(), which prevents any useful information regarding training data or model to the TRE.

Because of this, we are unable to proceed with this release, and the user is requested to call one of the above functions.

### User Story 8: User provides safemodel with no data

In this example, a user builds a model but does not use a SafeModel, and does not wrap their training/testing data in a Target object. The user only provides a description of the pre-processing which has been done.

Unfortunately, at this point, we cannot provide a recommendation to either release or reject the model. The researcher should be prompted to either wrap their data in a Target object, or provide a copy of their training and testing data.
