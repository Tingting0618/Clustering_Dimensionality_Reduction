# Dimensionality Reduction and Clustering

## Topic 1: PCA

Many Machine Learning problems involve thousands or even millions of features for each training instance. Not only do all these features make training extremely slow, but they can also make it much harder to find a good solution. This problem is often referred to as the curse of dimensionality.

High-dimensional datasets are at risk of being very sparse. The more dimensions the training set has, the greater the risk of overfitting it. There are two main approaches to reducing dimensionality: 
  - Projection: Projection is helpful when instances lie close to a lower-dimensional (2D) subspace.
  - Manifold Learning: Manifold Learning is helpful when the subspace may twist and turn, such as in the famous Swiss roll toy dataset.
    -  manifold assumption: high-dimensional datasets lie close to a much lower-dimensional manifold. 

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

## Topic 3: Anomaly Detection Using Gaussian Mixtures
- Anomaly detection (also called outlier detection) is the task of detecting instances that deviate strongly from the norm.
- Using a Gaussian mixture model for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly.

#### Step 1: Simulate 2 clusters:
![download](https://user-images.githubusercontent.com/44503223/126397269-e25baf38-9f5b-47fa-b4bf-96420d8cc0de.png)

#### Step 2: Using Gaussian Mixtures to identify cluster means, decision boundaries, and density contours
![download (1)](https://user-images.githubusercontent.com/44503223/126397363-bdfdc1f7-7ca4-4e52-8ca9-1e4efc2836c5.png)

#### Step 3: Anomaly Detection
- Using a Gaussian mixture model for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly.
- Identify the outliers using the fourth percentile lowest density as the threshold (i.e., approximately 0.5% of the instances will be flagged as anomalies)

![download (4)](https://user-images.githubusercontent.com/44503223/126397886-88c48bd5-93f2-4fbb-b871-1335a8c76084.png)

#### Step 4: Run iterations to select the most optimal cluster
![download (3)](https://user-images.githubusercontent.com/44503223/126397646-44652944-9757-4fc8-bb09-db370b043369.png)

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io) page.

## References:
This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
- StatQuest: https://statquest.org/
