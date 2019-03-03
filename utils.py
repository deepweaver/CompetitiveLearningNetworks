# import matplotlib.pyplot as plt 
from matplotlib import pyplot 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

def randomCentroids(dataset, centroids_num=2):
    '''
        return a random centroids matrix, each centroid should be 
        confined to the range of dataset of the same axis
    '''
    class_num = dataset.shape[1]
    centroids = np.zeros((centroids_num, class_num))
    for j in range(class_num):
        max_val = np.max(dataset[:,j])
        min_val = np.min(dataset[:,j])
        for i in range(centroids_num):
            # print(np.random.random())
            centroids[i,j] = min_val + np.random.random() * (max_val - min_val)
    return centroids 




def visualizeData(dataset, centroids=None, randcentroids=False):
    ''' numpy 3d array as input
        dataset.shape = (sample_num, class_num)
        centroids.shape = (centroids_num, class_num)
        if don't have centroids as input, you can set randcentroids as True to generate random centroids
    '''
    class_num = dataset.shape[1]
    fig = pyplot.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = dataset[:,0]
    sequence_containing_y_vals = dataset[:,1]
    sequence_containing_z_vals = dataset[:,2]
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    if centroids is not None:
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c='r', s=40)
    if centroids is None:
        if randcentroids == True:
            rand_centroids_num = np.random.randint(2,5)
            # centroids = np.random.rand(centroids_num, class_num)
            centroids = randomCentroids(dataset,rand_centroids_num) 
            ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c='r', s=40)
    pyplot.show()

def getData(filename="./dataset_noclass.csv"):
    '''
        read csv data
    '''
    file = open(filename).read()
    content = file.split('\r\n')[1:-1]
    sample_num = len(content)
    class_num = len(content[0].split(','))
    for i in range(sample_num):
        content[i] = map(float, content[i].split(','))
    dataset = np.array(content)
    return dataset, sample_num, class_num













if __name__ == "__main__":
    dataset, _, _ = getData()
    print(dataset.shape)
    visualizeData(dataset,randcentroids=True) # we can see from the plot that the 3d scatter points 


