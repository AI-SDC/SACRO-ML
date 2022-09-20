# Risk Examples

This area in the AI-SDC repository holds some notebooks that illustrate the _risk_ of disclosing trained ML models.  It consists of 5 notebooks (3 python, 2 R), each highlighting one aspect of vulnerability.

## Contents

### `python`

- **Finding out unknown information about a famous person (Attribute Inference)** 

`attribute_inference_cancer.ipynb` - An example of attribute inference that demonstrates how access to the model, plus some information about one of the participants (perhaps they are famous) can allow the inference of other, sensitive attributes.
- **Identifying if someone famous has suffered from cancer (Membership Inference)**
 
`membership_inference_cancer.ipynb` - An example (using the same dataset as in the attribute inference example) in which it is possible to infer someone's presence in the training data and therefore learn something sensitive about them (that they had cancer).
- **Hospital admission survival (instance-based model vulnerability)**
 
`instance_based_mimic.ipynb` - Demonstrating how an instance-based classifier (in this case a Support Vector Machine) stores exact copies of some rows from the training data.

### `R`

- **Finding private information from publicly available data**

`attribute_inference_old.Rmd` - An example to demonstrate that even very simple models (in this case, ordinary least squares) can, if fitted poorly, be disclosive. [Smarti: not in doc?]
- **Successful candidates in a job interview (Membership Inference)**

`membership_inference_solvency.Rmd` - An example of a membership inference attack that allows an attacker to infer that individuals were in the training set and hence that they were IV drug users.
