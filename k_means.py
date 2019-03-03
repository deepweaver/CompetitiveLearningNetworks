from ann import * 
from utils import * 
dataset, _, _ = getData()
epsilon = 0.001
eta = 0.7


# nn = ann(np.array([[1,0,0.1],[-1,2,0]]).T, np.array([[0,0],[0,1]]))
centroids = randomCentroids(dataset)
# print(centroids)
nn = ann(dataset, centroids)
delta = nn.k_m()
# print()
while np.min(delta) > epsilon:
    delta = nn.k_m()

print(nn.W1) 
visualizeData(dataset, centroids=nn.W1) 

# with open("./classes.csv",'w') as file:
#     obj = np.c_[dataset,nn.catag]
#     for i in range(obj.shape[0]):
#         file.write(','.join(map(str,obj[i,:])))
#         # for j in range(obj.shape[1]):
#         #     file.write(str(obj[i,j])+',')
#         file.write('\n')




