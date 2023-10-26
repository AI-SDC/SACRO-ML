# User Stories
In this section there are code examples of how the AI-SDC tools can be used by both a researchers or Trusted Research Environment (TRE) users and a TRE output checkers. Each project is unique and therefore how AI-SDC tools are applied may vary from case to case. The user guides have been split into 7 'user stories', each designed to fit a different use-case.

The following diagram is intended to identify the closest use-case match to projects:

![User Stories](user_stories_flow_chart.drawio.png)

## General description
The user stories are coding examples intended to maximise the chances of a successful and smooth Machine Learning (ML) model egress from the TRE. These guides are useful to create appropriate ML models and metadata files necessary for output checking of the ML model prior the egress. Saving time and effort, and ultimately optimising costs.

Each user story consists of at least 2 files:
  > - **user_story_[x]_researcher.[py/R]**  Example on how to generate a ML model for the TRE users/researchers.
  > - **user_story_[x]_tre.py**  Example on how to perform attacks and generate a report.

Extra examples on how to use [safemodels](https://github.com/AI-SDC/AI-SDC/tree/development/example_notebooks) and perform [attacks](https://github.com/AI-SDC/AI-SDC/tree/development/examples) can be found following the corresponding links.

## Programming languages

Although, AI-SDC tools is written in Python, some projects may use a different programming language to create their target model. However, where possible Python should be preferred as more extensive risk-disclosure testing has been performed.

While most of the stories are Python examples, `user_story_4` is written in R.

## Instructions

**For researchers or users**
1. Select the best use-story match to the project.
2. Familirise with the relevant user-stroy example, both for researchers and TRE. Understanding how the process work both sides will increase the changes of smooth project.
3. Pre-process data and generate the ML model as appropriate for the project inside the TRE. Remember to follow the researcher relevant user story example code (**user_story_[x]_researcher.[py/R]**).
4. Make sure you generated all metadata, data and files required for output checking.
5. Fill out the `default_config.yaml` with the appropriate fields. An example of this file can be found [here](https://github.com/AI-SDC/AI-SDC/blob/user_story_visibility/user_stories/default_config.yaml) with required experiment parameters.
6. Run the command `python generate_disclosure_risk_report.py`. This will create a **release_files** where the all the required files, data and metadata for egress are placed.

*Alternative to steps 5 and 6*

5. Create a new configuration file using the same format in the [default_config.yaml](https://github.com/AI-SDC/AI-SDC/blob/user_story_visibility/user_stories/default_config.yaml) file with a different name.
6. Run the command `python generate_disclosure_risk_report.py --config <your_config_file_name>`.

**For TRE output checkers**
1. Select the best use-story match to the project. This should have been agreed with the user/researcher beforehand preferably.
2. Familirise with the relevant user-stroy example, both for researchers and TRE. Understanding how the process work both sides will increase the changes of smooth project.
3. Once the researcher/user requests the ML model egress, perform attacks according to corresponding **user_story_[x]_tre.py** example.
4. Look at the reports produced and make a judgment for model egress.

## The user stories in detail

Unless otherwise specified, the stories are for Python ML models.

### User story 1: Ideal Case

The user is familiar with AI-SDC tools. Also, the ML classifiers chosen for the project has been wrapped with the SafeModel class. This class, as the name indicates, ensures that the most leaky ML hyperparameters cannot be set, and therefore reducing the risk of data leakage from the generated model. Moreover, the user created the `Target` object provided by AI-SDC tools. This ensures an easier process to generate the data and metadata files required for the model release.

The user can perform attacks to check the viability of their model for release and have the opportunity to make changes where necessary. Once the user is satisfied, and generated all the attacks, the TRE staff performs an output check and makes a decision. This way the user optimises the time for realease.


### User story 2: SafeModel class and Target object employed

The user is familiar with AI-SDC tools. Also, the ML classifiers chosen for the project has been wrapped with the SafeModel class. This class, as the name indicates, ensures that the most leaky ML hyperparameters cannot be set, and therefore reducing the risk of data leakage from the generated model. Moreover, the user created the `Target` object provided by AI-SDC tools. This ensures an easier process to generate the data and metadata files required for the model release.

In this example, the user does not use the function `request_release` and does not provided all the data and metadata files required for output check. Which means that the output checker has to recreate the processed data (code provided by user). The user also needs to state which rows of the data were used for training and testing of the model. Once all of this is fulfilled, the output checker can run attacks, generate reports and make a decision on model release.


### User Story 3: User provides dataset object but does not use SafeModel

There exist a vast number of classifiers available and only for a few the SafeModle wrapper exists. Therefore, for some purposes will not be possible to use the SafeModel class.

- In this case we don’t currently have any checking for TRE-approved hyper-parameters or for class disclosure.
  - But if it is a type where we have a safemodel version, we could create functionality to load it and then check hyper-parameters using existing code
  - This raises the issue of whether safeModelClassifiers should have a load() option ?? – Is currently commented out
  - Could also provide method for checking for k-anonymity (and possible pure nodes) where appropriate by refactoring safemodels.
- TREs need to manually configure and start scripts to do LIRA, Worst_Case and Attribute_Inference attacks
   - NB this assumes their classifier outputs probabilities.

### User Story 4: User does not use safeXClassifier, or provide dataset object
#### but does provide description of pre-processing, and provides output probabilities for the train and test set they have used
- We cannot assume that the TRE has the capability to get the right bits of pre-processing code from their source code.
- Do we insist on this (would be needed for ‘outside world’)? what if this is commercially sensitive?
- TRE can in theory run LIRA and worst-case but not attribute inference attacks.
  - There is a risk that they have misidentified the train/test splits to give us ones which make the classifier look less disclosive
  - But this probably falls outside our remit?
- Recommend reject???
-We could automate generalisation (as lower bound) and  worst case attacks if they give output probabilities
   – so we need to specify format
- TRE would need actual copies of processed data to run LIRA

**THIS would be the version that let people use R **

### User Story 5:  User creates differentially private algorithm (not via our code) and provides sufficient details to create data object.
##### Status: not yet implemented
- How do we know what the actual epsilon value is?
- If it is a keras model we can reload and query it if they have stored the training object as part of the model save (we need epochs, dataset size, L2 norm clip, noise values).
  -  But then their stored model probably has disclosive values in anyway …
  -   So would have to delete before release.
  - And anyway, are keras models safe against attacks that change ‘trainable’ to true for different layers and then do repeated queries viz, attacks of federated learning.
- If it is non keras then do, we take it on trust??
  - Probably yes that comes under safe researcher??

- TRE can recreate processed training and test sets and run attacks.
- Does the actual epsilon value matter if we are doing that?
   - Yes probably, because it is the sort of thing a TRE may well set as a policy.

### User Story 6: Worst Case
##### Status: not yet implemented
- User makes R model for a tree-based classifier that we have not experimented with.
- TREs get researcher to provide at minimum the processed train and test files.

- From those we can’t run LIRA (because what would shadow models be?)
-  but we can  worst-case from the command line or a script if their model outputs probabilities.
-  And we can measure generalisation error.
-  But not attribute inference.
- We have no way of checking against class disclosure e.g. all training items in a specific subgroup ending in a ‘pure’ node.

- Very hard to check and recommend release

### User Story 7: User provides safemodel with no data
- User loads in data and pre-processes out with Target object
- User uses SafeDecisionTreeClassifier
- User calls request_release() themselves, but does not pass data object to request_release() or save processed form of data.
 - SafeDecisionTreeClassifier report checks for class disclosure and TRE risk appetite for algorithm X.
- User may send the dataset to TRE, but does not provide details of pre-processing, nor gives details about which samples were used for training/testing
- TRE has to rely on their own judgement and what the researcher has told them - AISDC in this case cannot provide any additional assistance
