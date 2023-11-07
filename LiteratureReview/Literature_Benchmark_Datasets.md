1. A review paper published in 2022 - [Membership Inference Attacks on Machine Learning:
A Survey](https://dl.acm.org/doi/pdf/10.1145/3523273)
2. [Github repository](https://github.com/HongshengHu/membership-inference-machine-learning-literature) on extended literature by the first author of the review paper. This includes papers published in 2023.
3. Most frequent datasets used in MIA attacks (as per the review paper mentioned above)
   - Task: Classification
     |Dataset|Source Paper|Data Source|No of MIA studies|
     |---|---|---|---|
     |Purchase-100||https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data|27|
     |Adult||https://archive.ics.uci.edu/dataset/2/adult|17|
     |Texas-100||https://www.dshs.texas.gov/texas-health-care-information-collection/health-data-researcher-information/texas-inpatient-public-use|17|
     |Foursquare|https://dl.acm.org/doi/pdf/10.1145/2814575|https://foursquare.com/|11|
   - Classification/Generation Benchmarks
     |Dataset|Source Paper|Data Source|No of MIA studies|
     |---|---|---|---|
     |CIFAR-10|https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf|https://www.cs.toronto.edu/~kriz/cifar.html|40|
     |CIFAR-100|https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf|https://www.cs.toronto.edu/~kriz/cifar.html|30|
     |MNIST|https://ieeexplore.ieee.org/document/726791|http://yann.lecun.com/exdb/mnist/|30|
     |Fashion-MNIST|https://arxiv.org/pdf/1708.07747.pdf|https://github.com/zalandoresearch/fashion-mnist|10|
     |LFW|||9|
   - Classification (Graphs)
     |Dataset|Source Paper|Data Source|No of MIA studies|
     |---|---|---|---|
     |Citeseer|http://eliassi.org/papers/ai-mag-tr08.pdf||3|
     |Cora|http://eliassi.org/papers/ai-mag-tr08.pdf||3|
5. Metric used in the literature (as per the review paper mentioned above)
   |Name|Description|
   |---|---|
   |Accuracy||
   |Generalisation Error||
   |Attack Success Rate (ASR)|$\frac{\textnormal{No of Successful attacks}}{\textnormal{No of all attacks}}$|
   |Attack Precision (AP)|$\frac{\textnormal{No of members classified as members}}{\textnormal{No of records classified as members}}$|
   |Attack Recall (AR)|$\frac{\textnormal{No of members correctly classified as members}}{\textnormal{No of all members}}$|
   |Attack False Positive Rate (FPR)|$\frac{\textnormal{No of non-members classified as members}}{\textnormal{no of all non-members}}$|
   |Membership Advantage (MA)|$AR-FPR$|
   |Attack F1-Score|$\frac{2\times AP\times AR}{AP-AR}$|
   |Attack AUC|$\frac{\textnormal{}}{\textnormal{}}$|

