### User Story 8: User provides safemodel with no data

In this example, a user builds a model but does not use a SafeModel, and does not wrap their training/testing data in a Target object. The user only provides a description of the pre-processing which has been done.

Unfortunately, at this point, we cannot provide a recommendation to either release or reject the model. The researcher should be prompted to either wrap their data in a Target object, or provide a copy of their training and testing data.
