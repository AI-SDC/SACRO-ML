### User Story 4: User does not use safeXClassifier, or provide dataset object
#### but does provide description of pre-processing, and provides output probabilities for the train and test set they have used

In this example, a researcher has a model (written in Python or R for example) which makes a prediction based on some data. The researcher has not provided a copy of their training data, but has provided a list of output probabilities for each class their model predicts, for each sample in their dataset, in a .csv file format.

The TRE, by using this package and this user story, can run some of the attacks available in this package. Doing so will generate a report, which will help the TRE to make a decision on whether the model should be released.