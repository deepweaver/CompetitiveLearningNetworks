## Assignment 3 --- Implement Competitive Learning Networks
### Part I: ANN and Kohonen
1. Design a simple two layer competitive learning clustering ANN model to find clusters in the given dataset using unsupervised learning. Each output node should implement a linear output function (sum of weighted inputs) and should have inhibitory connection with all other output nodes to implement a Kohonen layer as discussed in the class. Multiple iterations may be needed at the Kohonen layer to find the winning node. (3 marks)
2. NexttrainthemodelweightsbasedonKohonenlearningalgorithmto find two clusters in the dataset such that the clustering error is minimum. Use appropriate terminating conditions. (2 marks)
3. Markswillbedeductedforinaccuratecodingandalgorithm.Code should be easy to understand and commented. You will also lose mark if code cannot be executed.

### Part 2: K-means
1. UsethesamenetworkbutimplementtheK-meansalgorithmthistime (replace the learning algorithm in the previous part) as discussed in the class. (2 marks)
>Tip: If you have multiple winning nodes for a data point, you can ignore updating the weight vector for that data point in that iteration – state your design choice in the text file.

### Part 3: Report
1. You must submit the following in the text file for each part of the assignment.
2. Explain your assumptions and design choices in a text file for both parts 1 and 2, and analyze your result. (1 mark)
3. Weight vectors of the cluster centres, the sum squared error for each cluster centre (and the total for both centres) for all data points in the cluster) for part 1 and 2. (2 marks)


### Requirement
- matplotlib
- numpy 

### Run
`python k_means.py` 
`python KohonenLearning.py`
both output 1. visulization of all the points and 2. weight matrix(centrods positions)

### Explanation 

#### utils.py

###### `def visualizeData(dataset, centroids=None, randcentroids=False)` 
this function visualize data, before training, the dataset visualization is saved as `dataset_3d_xy.png` and `dataset_3d_z.png`
after training, the centroids from kohonen learning and k means can be found in `Kohonen_centroids.png` and `k_means.png` respectively 
###### `def randomCentroids(dataset, centroids_num=2)`
1. `visualizeData(dataset, centroids=None, randcentroids=False)` calls this function to generate and visualizatio random centroids with red color
2. both algorithm requires random initialize weights
   
###### `def getData(filename="./dataset_noclass.csv")`
reads data, return a numpy array with shape (1000,3) 

#### ann.py
in the first layer, `def hammingnet(self):` calculate the eucilidian distances between each centorids and each training data, the output layer `def forward(self,)` calls `maxnet()` and return the index of training data that is the closest to centroids. This part is for kohonen learning
`def k_m(self,)` update the weight matrix to the new center, then return the absolute value delta weight, when the update delta is smaller than a certain value, the outer loop in `k_means.py` will top.

#### KohonenLearning.py
performs kohonen learning
```python
➜  assignment3 python KohonenLearning.py 
[[ 0.77096561  0.151495    1.18909862]
 [ 0.24621845  0.34526104 -0.12839828]]
➜  assignment3 python KohonenLearning.py 
[[-0.19023832  0.57666815  0.57469263]
 [ 1.21994888 -0.08828772  0.42141771]]
➜  assignment3 python KohonenLearning.py 
[[ 0.69248427  0.32728859 -0.68562296]
 [ 0.44474542  0.23316223  0.81332862]]
➜  assignment3 python KohonenLearning.py 
[[ 1.20852245 -0.08466805  0.43545362]
 [-0.20137777  0.5836541   0.56336277]]
➜  assignment3 python KohonenLearning.py 
[[ 0.707603   -0.26101248  1.73602012]
 [ 1.16696622  0.42918482  1.68369775]]
➜  assignment3 python KohonenLearning.py 
[[ 1.17053969 -0.06384953  0.5590387 ]
 [-0.23074274  0.59453752  0.43639638]]
➜  assignment3 python KohonenLearning.py 
[[ 1.21994888 -0.08828772  0.42141771]
 [-0.19023832  0.57666815  0.57469263]]
```
#### k_means.py
performs k means learning
```python
➜  assignment3 python k_means.py 
[[ 1.21690311 -0.08769355  0.40885214]
 [-0.19008959  0.57740057  0.58696855]]
➜  assignment3 python k_means.py 
[[ 1.22275388 -0.08121257  0.41859688]
 [-0.19290117  0.56995158  0.57737049]]
➜  assignment3 python k_means.py 
[[ 1.21814224 -0.09703706  0.48967379]
 [-0.18852325  0.58497405  0.50989592]]
➜  assignment3 python k_means.py 
[[-0.18976641  0.58027814  0.60153828]
 [ 1.2108239  -0.0879739   0.39441474]]
➜  assignment3 python k_means.py 
[[-0.20430205  0.58337149  0.58574329]
 [ 1.20297685 -0.08035713  0.41366388]]
➜  assignment3 python k_means.py 
[[-0.19003662  0.5762132   0.59097445]
 [ 1.21684752 -0.08644778  0.40464923]]
➜  assignment3 python k_means.py 
[[-0.19008959  0.57740057  0.58696855]
 [ 1.21690311 -0.08769355  0.40885214]]
```

#### classes.csv
the number of the last column shows which centroids this data point belongs to.