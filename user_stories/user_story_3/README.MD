# User story 3

# No SafeModel but Target object employed

- In this case we don’t currently have any checking for TRE-approved hyper-parameters or for class disclosure.
  - But if it is a type where we have a safemodel version, we could create functionality to load it and then check hyper-parameters using existing code
  - This raises the issue of whether safeModelClassifiers should have a load() option ?? – Is currently commented out
  - Could also provide method for checking for k-anonymity (and possible pure nodes) where appropriate by refactoring safemodels.
- TREs need to manually configure and start scripts to do LIRA, Worst_Case and Attribute_Inference attacks
   - NB this assumes their classifier outputs probabilities.