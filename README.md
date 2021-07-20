# Dimensionality Reduction and Clustering

## Topic 1: PCA

Principal Component Analysis (PCA) is by far the most popular dimensionality reduction algorithm. 
First it identifies the hyperplane that lies closest to the data, and then it projects the data onto it.
Scikit-Learn’s PCA classes take care of centering the data.

#### PCA uses singular value decomposition:
- step 0: plot the data
- step 1: get the center of the data
- step 2: shift the center to the plot center (0,0)
- step 3: start a random line that goes through the origin
- step 4: minimize the distance between a dot and the line
- step 5: the line is called PC1: a linear combinator of variables
- step 6: PC 2 is perpendicular to PC1
- step 7: rotate PC1 and PC2 so that PC2 is vertical and PC1 is horizontal

This process called Singular Value Decomposition 

#### Terminology: 
- The % of variables that consist of the PC 1 called Loading Score. 
- PC1 Variation = SS(distances for PC1)/(n-1)
- PC2 Variation = SS(distances for PC2)/(n-1)
- Total variation around PCs = PC1+PC2
- Scree plot: % of variations that each PC accounts for. 

## Topic 2: K-Means

The K-Means algorithm is a simple algorithm capable of clustering this kind of dataset very quickly and efficiently, often in just a few iterations.
- Euclidean distance
- Need to scale and center variable first (assuming all variables are equally important)

#### K-Means Concept:
- determine K (number of clusters)
- randomly select K points
- measure the distance between the first point and the first cluster
- assign the first point based on its nearest cluster
- calculate total variance within the clusters
- go back to step b, re-randomly assign 3 clusters.
- repeat, until the optimal clusters no longer change.

#### K-Means model evaluation metric
The mean squared distance between each instance and its closest centroid.

## References:
This repo is my learning notebook following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
- StatQuest: https://statquest.org/
