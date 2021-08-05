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

**Incremental PCA** allows us to split the training set into mini-batches and feed an IPCA algorithm one mini-batch at a time. This is useful for large training sets and for applying PCA online (i.e., on the fly, as new instances arrive).

**Kernel PCA** is good at preserving clusters of instances after projection, or sometimes even unrolling datasets that lie close to a twisted manifold.
- e.g., linear, RBF, sigmoid kernels. 

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

## Topic 2: Other Dimensionality Reduction Techniques
- Multidimensional Scaling (MDS)
  - Reduces dimensionality while trying to preserve the distances between the instances.

- t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize clusters of instances in high-dimensional space.

- Linear Discriminant Analysis (LDA)
  - LDA projection will keep classes as far apart as possible, so LDA is a good technique to reduce dimensionality before running another classification algorithm.
    - Max distance between means
    - Min variance within known groups

## Topic 3: K-Means

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
Inertia = The mean squared distance between each instance and its closest centroid.

#### K-Means model limitations
- It is necessary to run the algorithm several times to avoid suboptimal solutions
- Need to specify the number of clusters
- Not behave very well when the clusters have varying sizes, different densities, or nonspherical shapes.

## Topic 4: Anomaly Detection Using Gaussian Mixtures
- Anomaly detection (also called outlier detection) is the task of detecting instances that deviate strongly from the norm.
- Using a Gaussian mixture model for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly.

#### Step 1: Simulate 2 clusters:
![download](https://user-images.githubusercontent.com/44503223/126397269-e25baf38-9f5b-47fa-b4bf-96420d8cc0de.png)

#### Step 2: Using Gaussian Mixtures to identify cluster means, decision boundaries, and density contours
![download (1)](https://user-images.githubusercontent.com/44503223/126397363-bdfdc1f7-7ca4-4e52-8ca9-1e4efc2836c5.png)

#### Step 3: Anomaly Detection
- Using a Gaussian mixture model for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly.
- Identify the outliers using the lowest density as the threshold (i.e., approximately 0.5% of the instances will be flagged as anomalies)

![download (4)](https://user-images.githubusercontent.com/44503223/126397886-88c48bd5-93f2-4fbb-b871-1335a8c76084.png)

#### Step 4: Run iterations to select the most optimal cluster
![download (3)](https://user-images.githubusercontent.com/44503223/126397646-44652944-9757-4fc8-bb09-db370b043369.png)

## Topic 5: Hierarchical Clustering

**Logic:**
- Step1: figure out which variable is most close to var1
- Step2: figure out which variable is most close to var2
- Step3: of all those above combinations, figure out the closest group, cluster it
- Step4: go back to step1, but treat the cluster in Step3 as 1 new variable

**Definition of close:**

- Euclidean: sqrt(diff_sample1^2+diff_sample2^2+diff_sample3^2)
- Manhattan: abs(sample1_diff)+abs(sample2_diff)+abs(sample3_diff)

**Steps:**
- Compare 1 new observation to the existing cluster (multiple variables) 
- Compare that 1 vs the average of the cluster - centroid 
- Compare that 1 vs the closest variable in that cluster - single-linkage
- Compare that 1 vs the furthest variable in that cluster - complete-linkage

```r
## NOTE: You can only have a few hundred rows and columns in a heatmap
## Any more and the program will need to much memory to run. Also, it’s
## hard to distinguish things when there are so many rows/columns.

## Usually I pick the top 100 genes of interest and focus the heatmap on those
## There are, of course, a million ways to pick 100 "genes of interest".
## You might know what they are by doing differential gene expression
## you might just pick the top 100 with the most variation across the
## samples. Anything goes as long as it helps you gain insight into the data.

## first let's "make up some data"...

sample1 <- c(rnorm(n=5, mean=10), rnorm(n=5, mean=0), rnorm(n=5, mean=20))
sample2 <- c(rnorm(n=5, mean=0), rnorm(n=5, mean=10), rnorm(n=5, mean=10))
sample3 <- c(rnorm(n=5, mean=10), rnorm(n=5, mean=0), rnorm(n=5, mean=20))
sample4 <- c(rnorm(n=5, mean=0), rnorm(n=5, mean=10), rnorm(n=5, mean=10))
sample5 <- c(rnorm(n=5, mean=10), rnorm(n=5, mean=0), rnorm(n=5, mean=20))
sample6 <- c(rnorm(n=5, mean=0), rnorm(n=5, mean=10), rnorm(n=5, mean=10))

data <- data.frame(sample1, sample2, sample3, sample4, sample5, sample6)
head(data)

## draw heatmap without clustering
heatmap(as.matrix(data), Rowv=NA, Colv=NA)

## draw heatmap with clustering and use default settings for everything
##
## The heatmap function defaults to using…
## The Euclidean distance to compare rows or columns.
## "Complete-linkage" to compare clusters.
##
## Also, heatmap defaults to scaling the rows so that each row has mean=0
## and standard deviation of 1. This is done to keep the range of colors
## needed to show differences to a reasonable amount. There are only so
## many shades of a color we can easily differentiate…
heatmap(as.matrix(data))

## draw heatmap with clustering and custom settings…
## We’ll use the ‘manhattan’ distance to compare rows/columns
## And "centroid" to compare clusters.
## The differences are… subtle…
heatmap(as.matrix(data),
distfun=function(x) dist(x, method="manhattan"),
hclustfun=function(x) hclust(x, method="centroid"))

## Now do the same thing, but this time, scale columns instead of rows…
heatmap(as.matrix(data),
distfun=function(x) dist(x, method="manhattan"),
hclustfun=function(x) hclust(x, method="centroid"),
scale="column")

## if you want to save a heatmap to a PDF file….
pdf(file="my_first_heatmap.pdf", width=5, height=6)
heatmap(as.matrix(data))
dev.off()
```

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io) page.

## References:
This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
- StatQuest: https://statquest.org/statquest-hierarchical-clustering/

