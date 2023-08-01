## User story 1: Ideal Case
- User creates an object “mydata” of type aisdc.attacks.dataset.Data and provides a separate code file that does the translation between the data in the format provided and the data in the format to be input to the machine any model.
- User creates a model “mymodel” from the safeXClassifier class and calls mymodel.fit().
- User calls mymodel.preliminary_check() to make sure their hyper-parameters are within the TRE risk appetite for algorithm X.
- User calls mymodel.run_attack(mydata) for different attack types and iterates over different hyper-parameters until they have an accurate model, and they interpret attack results as safe.
- User calls myModel.request_release() with parameters modelsavefile.sav and  again passing the mydata object (without it request_release does not run attacks).
  - LIRA, worst_case, and attribute_inference attacks are run automatically,
  - results are stored ready for the TRE output checkers to look at.
  - System also saves the results of mymodel.posthoc_check() for poor practice, model edits etc.
- TRE checker has everything they need to make a decision with no further processing.
