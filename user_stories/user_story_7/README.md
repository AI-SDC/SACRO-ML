## 7: User provides safemodel with no data
- User loads in data and pre-processes out with Target object
- User uses SafeDecisionTreeClassifier 
- User calls request_release() themselves, but does not pass data object to request_release() or save processed form of data.
- SafeDecisionTreeClassifier report checks for class disclosure and TRE risk appetite for algorithm X.
- User may send the dataset to TRE, but does not provide details of pre-processing, nor gives details about which samples were used for training/testing
- TRE has to rely on their own judgement and what the researcher has told them - AISDC in this case cannot provide any additional assistance