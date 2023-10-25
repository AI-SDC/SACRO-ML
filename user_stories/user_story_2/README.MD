# User strory 2

## SafeModel and Target object only

- User provides Target object and code, uses safeXClassifier() but does not pass data object to request_release() or save processed form of data.
- safeXClassifer report checks for class disclosure and TRE risk appetite for algorithm X.
- TRE output checker has to manually recreate processed data using code provided.
- TRE output checker is unable to run any attacks UNLESS they also know exactly which rows from the dataset were used for training and testing.
- So dataset object needs to store those specific details OR use fixed values for seed (e.g. to sklearn.train_test_split() ) and be extremely transparent about how stratification was done.
- If TRE has enough info to recreate train/test processed data, then they can
    - Run attacks from script.
    - Then the post-processing script
    - Then make a judgement.