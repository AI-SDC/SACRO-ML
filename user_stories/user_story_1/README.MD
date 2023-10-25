# User story 1

## Ideal case

- User creates an object "target" of type aisdc.attacks.target.Target and provides a separate code file that does the translation between the data in the format provided and the data in the format to be input to the machine any model.
- User creates a model "model" from the safeXClassifier class and calls model.fit().
- User calls model.preliminary_check() to make sure their hyper-parameters are within the TRE risk appetite for algorithm X.
- User calls model.run_attack(target) for different attack types and iterates over different hyper-parameters until they have an accurate model, and they interpret attack results as safe.
- User calls model.request_release() with parameters modelsavefile.sav and  again passing the target object (without it request_release does not run attacks).
  - LIRA, worst_case, and attribute_inference attacks are run automatically,
  - results are stored ready for the TRE output checkers to look at.
  - System also saves the results of model.posthoc_check() for poor practice, model edits etc.
- TRE checker has everything they need to make a decision with no further processing.