### K_nearest_neighbours
K-Nearest Neighbours is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining and intrusion detection.

It is widely disposable in real-life scenarios since it is non-parametric, meaning, it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data).

We are given some prior data (also called training data), which classifies coordinates into groups identified by an attribute.

As an example, consider the following table of data points containing two features:

![k-nearest-neighbours1](https://user-images.githubusercontent.com/19835029/27906524-85cac24a-6261-11e7-858a-5cc232b1a68a.png)

Now, given another set of data points (also called testing data), allocate these points a group by analyzing the training set. Note that the unclassified points are marked as ‘yellow’.
![k-nearest-neighbours2](https://user-images.githubusercontent.com/19835029/27906540-99b4abe0-6261-11e7-8041-b0de9ca3c049.png)

# Intuition
If we plot these points on a graph, we may be able to locate some clusters, or groups. Now, given an unclassified point, we can assign it to a group by observing what group its nearest neighbours belong to. This means, a point close to a cluster of points classified as ‘Red’ has a higher probability of getting classified as ‘Red’.

Intuitively, we can see that the first point (2.5, 7) should be classified as ‘Blue’ and the second point (5.5, 4.5) should be classified as ‘Red’.

# Algorithm

Let m be the number of training data samples. Let p be an unknown point.

1.  Store the training samples in an array of data points arr[]. This means each element of this array represents a tuple (x, y).
2.  for i=0 to m:
  Calculate Euclidean distance d(arr[i], p).
3.  Make set S of K smallest distances obtained. Each of these distances correspond to an already classified data point.
4.  Return the majority label among S.

# Distance Measures One Can Take
Type of distance measure to use depends upon the experience or type of data to be processed. Some frequently used distance fucntions are

![k-nearest-neighbours3](http://www.saedsayad.com/images/Clustering_distance.png)
