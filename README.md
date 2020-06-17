# DTI
This repository contains the code of a drug target discovery approach.
This approach combines network embedding and machine learning to find potential interactions of drugs and proteins. 
It consists of two main steps:

* **Network Embedding:** In this step we constructed a vectorial reprensentation for drugs and proteins using compact feature learning that consists of:

     *  [Random Walk with Restart](diffusionRWR.py)
     *  [Diffusion Component Analysis](DCA.py)

     \- We then obtain a matrix reprenstation of drugs and proteins.
* **Edge Feature Learning:** We used 4 binary operators to learn [feature reprensation](edge_feature_learning_classification.ipynb) of each pair of drugs and proteins using their vectorial representation. Then we apply some machine learning classification algorithms on the learned representations to predict the potential connections between drugs and proteins.

The data that we used in this project can be found [here](data/).
