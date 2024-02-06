# User story 1

## Ideal case

The user is familiar with AI-SDC tools. Also, the ML classifiers chosen for the project has been wrapped with the SafeModel class. This class, as the name indicates, ensures that the most leaky ML hyperparameters cannot be set, and therefore reducing the risk of data leakage from the generated model. Moreover, the user created the `Target` object provided by AI-SDC tools. This ensures an easier process to generate the data and metadata files required for the model release.

The user can perform attacks to check the viability of their model for release and have the opportunity to make changes where necessary. Once the user is satisfied, and generated all the attacks, the TRE staff performs an output check and makes a decision. This way the user optimises the time for release.
