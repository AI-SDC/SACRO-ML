Glossary and definitions
========================

Actors
--------
*Trusted research environment:*
    A trusted research environment (TRE) sometimes called a ‘data enclave’, ‘research data centre’, 'secure data environments (SDEs)' or ‘safe haven’ is an analytical environment where the researchers working on the data have substantial freedom to work with detailed row-level data (see below) but are prevented from importing or releasing data without permission, and typically are subject to a monitoring and a significant degree of access control.

*Researchers:*
    Someone who has permission to use a TRE and has access to data within that TRE. For this work, researchers are assumed to be interested in building machine learning (ML) models that they will then wish to remove from the TRE. We use the term “researchers” to mean researchers who could be academic or from a commercial or government setting – all of whom could be training ML models to develop a solution in the public interest.

*Data controller:*
    A person(s) or corporate body, who alone or jointly with others determine the purposes for which and the manner in which any personal data are, or are to be processed (or who are the controller(s) by virtue of the Data Protection Act 2018, section 32(2)(b) (by means by which it is required by an enactment to be processed)), and all of whom are officially registered on the Information Commissioners Office Data Protection Public Register. Data controllers need to approve the use of their data for research projects. In this document, we are using the term Data Governance Committee to represent the group/person by which Data Controller(s) approve applications to use their data for the research project. TREs can be a service which is run by the same organisation that is the Data Controller or can be a service which is run by a separate organisation to the Data Controller. We have separated the roles of the Data Controller and TRE within these recommendations.

*Joint Data Controller:*
    A person(s) and/or body corporate, who jointly determine the purposes for which and how any personal data are or are to be processed (or who are the controller(s) under the Data Protection Act 2018, section 32(2)(b) (by means by which it is required by an enactment to be processed)) and all of whom are officially registered on the Information Commissioners Office Data Protection Public Register.

*Attacker or adversary:*
    A person or group of persons who attempts to extract, from the trained ML model, some or all of the personal data that was used to train it.

*End User:*
    A person who uses the ML model outside the TRE by obtaining access to the model either through access to a resource storing the code (e.g. a version control system such as GitHub), software that implements the model, or a web service that allows the model to be queried. This person is often a different person from the researchers who trained the model.

Data
--------
*Dataset:*
    Any collection of data about a defined set of entities. This usually refers to data where data units are distinguishable (ie not summary statistics).

*Aggregate-level data:*
    Summary data which is acquired by combining individual-level data and may be collected from multiple sources and/or on multiple measures, variables, or individuals. It is synonymous with **'aggregate(d) data’** or **‘statistical results’**.

*Personal data:*
    'Any information relating to an identified or identifiable living individual' as per section 3 of the UK Data Protection Act 2018 (which implements the EU’s General Data Protection Regulation or GDPR). Personal data is broadly interpreted, especially via the concept of ‘identifiable’ which is further defined as 'a living individual who can be identified, directly or indirectly in particular by reference to (a) an identifier such as a name, an identification number, location data or an online identifier, or (b) one or more factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity of the individual.

*Special categories of personal data:*
    The term used in current UK and EU data protection legislation for certain more sensitive categories of personal data. The special categories are: race; ethnic origin; political opinions; religious or philosophical beliefs; trade union membership; genetic data; biometric data (where this is used for identification purposes); health data; data about sex life; and sexual orientation. There are additional legal requirements which must be met to process special category personal data.

*Synthetic data:*
    Data artificially generated to replicate statistical properties of a given real-world dataset, ideally not containing genuine identifiable information i.e. no personal data.

*Row-level data:*
    A data set whose columns are different types of measurements/images/features, and where each row contains the record of an individual or organisation. It is synonymous with **‘record-level data’** or **‘microdata’**, and in contrast to aggregate-level data.

*Pseudo-anonymous dataset:*
    A data set in which any information which could be used to identify an individual has been replaced with a pseudonym.

*Anonymisation:*
    The process of rendering data in such a way that the people the data relates to are not, or are no longer, identifiable.

*Anonymised dataset:*
    A dataset where individuals are no longer identifiable. It is important to note that a person does not have to be named in order to be identifiable.

Machine learning
----------------
*Artificial intelligence (AI):*
    The theory and development of computer systems able to perform tasks normally requiring human intelligence.

*Pre-processing:*
    The process of transforming data prior to using it for training a statistical model.

*Training data:*
    Are collections of examples or samples that are used to 'teach' or 'train the machine learning model. The model uses a training data set to understand the patterns and relationships within the data, thereby learning to make predictions or decisions without being explicitly programmed to perform a specific task.

*Test data:*
    Is a separate sample, an unseen data set, to provide an unbiased evaluation of a model fit. The data should be of the same distribution. The test data set mirrors real-world data the machine learning model has never seen before. Its primary purpose is to offer a fair and final assessment of how the model would perform when it encounters new data in a live, operational environment. SACRO-ML uses of test data to simulate attacks to evaluate the potential of an attacker’s success.

*Validation data:*
    Contain different samples to evaluate trained ML models. It is still possible to tune and control the model at this stage. Working on validation data is used to assess the model performance and fine-tune the parameters of the model. This becomes an iterative process wherein the model learns from the training data and is then validated and fine-tuned on the validation set. A validation dataset tells us how well the model is learning and adapting, allowing for adjustments and optimizations to be made to the model's parameters or hyperparameters before it's finally put to the test.

*Algorithm:*
    A set of instructions to execute a task. This is a very general definition; algorithms may be deterministic (always giving the same answer when presented with the same input) or stochastic (giving different answers with various probabilities). Algorithms are not necessarily run by a computer; humans also use algorithms implicitly when making decisions. We will usually use the term to mean a set of instructions which only refer to data generically, rather than a specific dataset. An example of an algorithm is the ordinary least squares (OLS) method for fitting linear models.

*Machine Learning (ML) model:*
    Some computer code which implements an algorithm that, when presented with some input data, processes it in some way, and produces some output. There are many possible ML models, differing by the particular type of process they implement, and how they implement that process. For example, a particular ML model might process images (the input data) to assign them into a category (the output), which might be useful when attempting to build an ML system for diagnosing disease from a medical image.
Formally, a model is a set of candidate distributions over the domain of a given dataset (we leave differentiation between classical statistical models and ML models unspecified at present: there is no general distinction, and whether a model constitutes an ML model is best determined on a case-by-case basis).

*Trained Machine Learning (ML) model:*
    ML models are not usable until they have been trained. Training involves presenting the model with data that is relevant to the task at hand and modifying any parameters within the model to optimise its performance in the task of interest. For example, an ML model that is to be used for diagnosing breast tumours from mammograms will be trained with mammograms (input) with known tumour status (output). The training process will modify the parameters within the ML model such that the number of mistakes it makes on this “training” data is minimised. Once trained, the ML model can be used to generate an output for inputs that were not part of the training data: for example, to predict the tumour status for a new mammogram or predictive text on a smartphone.
Formally, a trained ML model is one of the candidate distributions of an ML model.

*Predictions:*
    The usable output of an ML model when given some data. Typically, this is the estimated chance of something happening given a set of inputs, where the estimation is made by the model. In the example above, the prediction would be the chance that the mammogram shows a real tumour.
Formally, a prediction is a (summary of a) conditional distribution derived from a trained ML model.

*Features:*
    Independent variables, often organised in columns in a given dataset used to train the model; e.g. age, sex, medical history, heart attack incidence.

*Target variable:*
    The outcome that an AI system seeks to predict.

*Target model:*
    An ML model (untrained, trained or being trained) that is the target of an attack.

*Instance-based models:*
    Also known as 'lazy-learners', are models which, to be able to make predictions, must ‘remember’ one or more training data samples exactly, rather than just summary data. These provide an immediate security risk, since specifying the model entails specifying individual samples. Such models are sometimes able to be made private by transforming training data samples randomly and only remembering the transformed samples.

    This include Support Vector Machines (SVMs) for example Support Vector Classifiers and Support Vector Regressors, Radial Basis Function Networks, k-Nearest Neighbours, Case-based reasoning, kernel models- alternative name given to a broad class which includes SVMs, Self Organising Map (SOM), Learning Vector Quantization (LVQ), Locally Weighted Learning (LWL).

*Ensemble methods:*
    The use of multiple methods, usually with their outputs combined through some form of the voting process, done to improve overall performance. This may involve the use of the same method on different parts of the dataset (e.g., random forests gradient boosting methods (e.g. XGBoost)) or different methods applied to the same dataset (e.g., super-learners).

*Kernel-based methods:*
    Group of model types that are used for pattern analysis. They use similarities between observations to build the model rather than the observations themselves. They are almost always instance-based methods, meaning that at least some of the training data must be saved within the trained model.

*Machine Learning (ML) model architecture:*
    The ML workflow specifies the various layers processes involved in the machine learning cycle: data acquisition, data processing, model engineering, execution and deployment. Broad categories of architecture Machine Learning are supervised learning, unsupervised learning, and reinforcement learning. Within each category, the architecture specifies the learning algorithm (e.g. neural networks, random forests, etc.) and its internal structure (e.g., number and type of layers in a neural network). A trained ML model is saved to a computer-readable file. Such a file could be loaded and used to make predictions or loaded to inspect the properties of the model.

*Hyper-parameters:*
    High-level parameters that can control aspects such as the model architecture (number of layers in a neural network, maximum depth of a decision tree, etc) and the learning process through which one particular trained model is chosen from all the possibilities

*Generalisation:*
    The ability of a machine learning model to make predictions on data that it did not see during training.

*Overfitting:*
    Situation in which a model fits and remembers the training data too well and does not generalise well for unseen data. Overfitting can facilitate membership attacks. Typically, small or unrepresentative training datasets can lead to overfitted models, especially if the data points have many features. A bad choice of hyper-parameters can also lead to overfitting (for example, excessively big neural network for simple classification tasks). Detection of subtle overfitting is difficult and a fundamental area of ML theory. More egregious overfitting can be readily identified by non-experts.

    Methods to reduce overfitting include: increasing the training dataset size, possibly using data augmentation techniques; using ‘regularisation’ techniques during training, which penalise candidate models for complexity; and optimising the choice of hyper-parameters, possibly with cross-validation. In neural networks, it is often beneficial to include dropout layers, which randomly deactivate neurons during training (effectively making the training procedure noisier). Differentially private optimizers (such as DP-SGD) add noise during the optimization steps/training process and may lead to better generalization.

*Data augmentation techniques:*
    Generate training samples from existing samples. In the case of images, a typical technique is to resize and rotate images in the original training set to generate new samples.

*Federated learning:*
    A technique that allows a machine learning algorithm to be trained on data that is stored in a variety of servers, devices, or TREs. The trained algorithm parameters (not data) are pooled into a central device which aggregates all individual contributions into a new composite algorithm.

*Disclosure control:*
    Methods to reduce the risk of disclosing information on the sensitive information (natural persons, households, economic operators and other undertakings, referred to by the data), usually based on restricting the amount of, or modifying, the data released.

*Data breach:*
    A breach of security leading to the accidental or unlawful destruction, loss, alteration, unauthorised disclosure of, or access to, personal data. This means that a breach is more than just losing personal data.

Disclosure risks
----------------
*Disclosure control methods:*
    Methods for reducing identification risk, usually based on restricting the amount of, or modifying, the data released.

*Disclosure risk:*
    The probability that a motivated intruder identifies or reveals new information, or both, about at least one person in disseminated data. Because anonymisation is difficult and has to be balanced against data utility, the risk that a disclosure will happen will never be zero. In other words, there will be a remote risk of identification present in all useful anonymised data.

*Disclosure:*
    The act of making data available to one or more third parties.

*Membership Inference:*
    The risk that an attacker (of either a White or a Black box) can create systems that identify whether a given data point was part of the data used to train the released model. This risk is far more likely to be disclosive of special category personal data in cases of medical data (X was part of a trial for a new cancer drug) than it is for other forms of data TREs might hold (Y was part of a survey on educational outcomes).

*Membership Inference Attacks (MIA):*
    A type of attack where an adversary wants to predict whether row data, which belongs to a single individual, was included in the training data set of the target model.

*Attribute Inference:*
    The risk that an attacker, given partial information about a person, can retrieve values for missing attributes in a way that gives them more information than they could derive just from descriptions of the overall distribution of values in the dataset.

*Attribute Inference Attacks (AIA):*
    A type of attack where the adversary is capable of discovering a few characteristics of the training data.

*Individual Disclosure:*
    Occurs when outputs from an analysis segment a participant with a specific condition, e.g. rare genetic disease, or a unique combination of conditions that might put the data of this individual at high risk of being identified or disclosed.

*Group (class) Disclosure:*
    Occurs where information about a group has been uncovered, and an individual can be identified as a member of that group; for example, the model might show that all males reporting for treatment aged 45-55 show traces of cocaine use.

*Disclosure by differencing:*
    Occurs when two separate outputs from a TRE can be used to infer private information by comparing them to each other, even if neither output allows such inference on its own. For example, given a fixed set of patients, if we fit one model to predict heart attack risks and another to predict lung cancer risk and release both, then we may be able to learn about the patients by comparing predictions on both models, even if we could not learn anything private from looking at only one of the models.

Disclosure mitigation related
-----------------------------

*Data Protection Impact Assessment (DPIA):*
    Is a process designed to identify risks arising out of the processing of personal data and to minimise these risks as far and as early as possible.

*Data Sharing Agreement:*
    Data sharing agreements set out the purpose of the data sharing, cover what happens to the data at each stage, set standards and help all the parties involved in sharing to be clear about their roles and responsibilities. Having a data sharing agreement in place helps demonstrate accountability obligations under the UK GDPR.

*Safe Wrapper:*
    Code that unobtrusively augments the functionality of existing software for machine learning. Typically, when a safe wrapper is applied, the model will retain the 'look and feel’ of its original version whilst adding functionality to:
    * Automate the running of various attacks to assess the vulnerability of a trained model.
    * Assist researchers in meeting their responsibilities, by warning when their choices for hyper-parameters or components are likely to result in models that are vulnerable to attack - and make suggestions for alternative choices.
    * Detect when researchers have either maliciously or inadvertently changed important parts of a model (or hyper-parameters) between training and requesting release.
    * Produce reports for TRE output checking staff summarising the above, to assist them in making good decisions about whether to release trained models.

*Differential privacy:*
    A mathematical framework that quantifies the privacy loss resulting from the inclusion of a person's data in a dataset. It ensures that the impact of any single record on the overall privacy is limited.


Other
-----

*GitHub:*
    An open online platform that lets people work collaboratively on projects/software codes from anywhere while tracking and managing changes to software code.

*Encryption:*
    A process which protects personal information by scrambling the readable text into incomprehensible text which can only be unscrambled and read by someone who has access to a specific decryption key.

*Public release:*
    This means making the content of a work public through publication, presentation, broadcast or other means.

*Release:* (when referring to a model)
    To export a trained machine learning model outside the (safe) environment (often called Safe Haven or TRE) for deployment and for making predictions. Another term for this is “egress”.

*Grant of a License or Transfer:* (when referring to a model)
    To release a model to specific researchers, by way of the grant of a license. A license is legally and contractually binding.

*Deploy:* (when referring to a model)
    To set up the trained model within an environment where it can be efficiently used to make predictions.

*License:* (when referring to a model)
    A license is a document that provides legally binding guidelines for the use and distribution of the AI/ML model.

    Typically provides end users with the right to one or more copies of the software without violating copyrights. The license also defines the responsibilities of the parties entering into the license agreement and may impose restrictions on how the model can be used.

*Reproducibility:*
    Means achieving a high degree of reliability or similar results when the study/experiment/ statistical analysis of a dataset is replicated.

*Model Disclosure Control (MDC):*
    Controls on privacy achieved exclusively through controlling aspects of the trained ML model, under the assumption that unlimited prediction queries may be made using the model by an attacker.

*Model Query Control (MQC):*
    Controls on privacy achieved by restricting access to or use of the trained ML model after release.



sources: GRAIMatter, https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/glossary/, https://www.techtarget.com/searchcio/definition/software-license
