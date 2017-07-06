#importing packages

import numpy as np
from matplotlib import pyplot as plt
import pandas as  pd

#caluclating euclidean distance
def distance(x1,x2):
    d=np.sqrt(((x1-x2)**2).sum())
    return d


#reading mnist data
df = pd.read_csv('mnist_train.csv')

#first 5 values of data
df.head()

#taking first 5000 values
data=df.values[:5000]
print (data.shape)


#splitting data into training and testing for validations & predicitions
split=int(0.8*data.shape[0])

x_train=data[:split,1:]
x_test=data[split:1:]

y_train=data[:split,0]
y_test=data[split:,0]


#k_nearest_neighbour function
def knn(x_train,y_train,xt,k):
    #empty list of values
    vals=[]
    
    #calculating and appending eulidean distance and appending to the list 
    for ix in range(x_train.shape[0]):
        d=distance(x_train[ix],xt)
        vals.append([d,y_train[ix]])
        
    
    #sorting the values based upon the first value in the "vals" list
    sorted_labels=sorted(vals, key=lambda z:z[0])
    #converting into numpy array
    neighbours=np.asarray(sorted_labels)[:k,-1]
    
    #getiing frequencies
    freq=np.unique(neighbours,return_coutns=True)
    return freq[0][freq[1].argmax()]

#plotting the figure and reshaping it to a size of 28x28 in grayscale
plt.figure(0)
plt.imshow( x_train[180].reshape((28,28)),cmap="gray")
print(y_train[0])
plt.show()


#defining the function for getting accuracy
def get_acc(kx):
    preds = []
    # print kx
    for ix in range(x_test.shape[0]):
        start = datetime.datetime.now()
        preds.append(knn(x_train, y_train, x_test[ix], k=kx))
        # print datetime.datetime.now() - start
    preds = np.asarray(preds)

    # print preds.shape
    return 100*float((y_test == preds).sum())/preds.shape[0]

print (get_acc(kx=15))
