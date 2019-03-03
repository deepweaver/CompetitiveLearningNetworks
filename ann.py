import numpy as np 
import copy 
class ann(object):
    def __init__(self, x, centroids):
        self.W1 = centroids 
        self.x = x.T
    def hammingnet(self):
        self._output = np.sum(self.W1**2, axis=1, keepdims=True) + np.sum(self.x**2, axis=0, keepdims=True) # (2,1) + (1,1000) = (2,1000)
        self._output -= 2 * np.dot(self.W1, self.x) 
        self._output = np.sqrt(self._output) 
        return self._output  
    def maxnet(self,arr):
        return np.min(arr, axis=1, keepdims=True), np.argmin(arr, axis=1)
    def forward(self,): # competitive layer
        self.output, self.outidx = self.maxnet(self.hammingnet())
        return self.output, self.outidx 
    def k_m(self,):
        self.hammingnet()
        self.catag = np.argmin(self._output, axis=0)
        # print(self.catag) 
 
        # print(self.x[:,self.catag==0].shape)
        # print(self.W1.shape)
        delta = copy.deepcopy(self.W1) 
        self.W1[0] = np.mean(self.x[:,self.catag==0], axis=1)
        self.W1[1] = np.mean(self.x[:,self.catag==1], axis=1)
        delta = np.abs(self.W1 - delta )
        return delta






if __name__ == "__main__":
    nn = ann(np.array([[1,0,0.1],[-1,2,0]]).T, np.array([0,0,0,1]).reshape(-1,2))
    result = nn.forward() 
    print(result)




