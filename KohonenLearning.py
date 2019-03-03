from ann import * 
from utils import * 
np.random.seed(3)
dataset, _, _ = getData()
epsilon = 0.001
eta = 0.7


# nn = ann(np.array([[1,0,0.1],[-1,2,0]]).T, np.array([[0,0],[0,1]]))
nn = ann(dataset, randomCentroids(dataset))

dW = np.zeros(nn.W1.shape) 
mindis, mindisidx = nn.forward()
while mindis.any() > epsilon:
    mindis, mindisidx = nn.forward() 
    
    # print(mindisidx.shape)
    # print(dataset[0].shape)
    # print(nn.W1.shape)
    for i in range(dW.shape[0]):
        idx = mindisidx[i]
        dW[i] = eta * (dataset[idx] - nn.W1[i])
    nn.W1 += dW

print(nn.W1) 
visualizeData(dataset, centroids=nn.W1) 


# print(result)