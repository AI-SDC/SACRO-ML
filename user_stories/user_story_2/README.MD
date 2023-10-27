# User story 2

## SafeModel class and Target object employed

The user is familiar with AI-SDC tools. Also, the ML classifiers chosen for the project has been wrapped with the SafeModel class. This class, as the name indicates, ensures that the most leaky ML hyperparameters cannot be set, and therefore reducing the risk of data leakage from the generated model.

In this example, the user does not use the `Target` object provided by AI-SDC tools, and does not call the function function `request_release` which provides all the data and metadata files required for output check. This means that the output checker has to recreate the processed data (code provided by user). The user also needs to state which rows of the data were used for training and testing of the model. Once all of this is fulfilled, the output checker can run attacks, generate reports and make a decision on model release.