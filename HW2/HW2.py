#importing libraries

%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#reading training data
dataset = pd.read_csv('1644871518_6273746_cleveland-train.csv')
dataset.head(152)

#processing train data
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
m,n= x.shape
print('x[0]={}'.format(x[0]))
print('y[0]={}'.format(y[0]))

#plotting some features of train data with respect to label

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0],x[:,5],y)
plt.show()

#implementing the inbuilt model for comparison
from sklearn.linear_model import LogisticRegression 
lr_model = LogisticRegression()
lr_model.fit(x,y)

y_pred_sk = lr_model.predict(x)

plt.clf()
plt.scatter(x[:,1],y)
plt.scatter(x[:,1], y_pred_sk, c="red")
plt.show()
print(f"Accuracy = {lr_model.score(x, y)}")


#defining functions required to implement gradient descent

def sigmoid(z):
  return 1/(1+np.exp(-z))

def lr_hyp(x, w):
  return np.dot(x,w)

z=np.ones(m)
z=z.reshape(m,1)
x=np.append(z,x,axis=1)
print('x[0]={}'.format(x[0]))
print(x.shape)

m,n=x.shape
w=np.zeros(n)
w = w.reshape(n,1)
y=y.reshape(-1,1)

iterations= 1000000
eta = 0.000001


#inorder to check log loss use 0 instead of -1 in the label
y = np.where(y==-1,0,y)

#log-loss or cross entropy 
def lr_cost(w,x,y):
  lfun = -y*(np.log(sigmoid(lr_hyp(x,w)))) - (1-y)*(np.log(1-sigmoid(lr_hyp(x,w))))
  l= np.sum(lfun)/m
  return l

Loss= lr_cost(w,x,y)
print(Loss)

#normalize function that subtracts mean and divides the feature set by standard deviation
def normalize(X):
    return (X - X.mean())/np.std(X)

#implementing gradient descent 
import time
def grad_descent(x,y,w,eta,iterations):
  start = time.time()
  for p in range(iterations):
    y_pred = sigmoid(lr_hyp(x,w))
    cfun1 = y_pred - y
    g=np.dot(x.transpose(),cfun1)/m
    if g.all()<0.01:
      break
    w = w - (eta*g)
  run_time = time.time() - start
  print(run_time)
  print(p)
  return w
w = grad_descent(x,y,w,eta,iterations)


#Predicting the output
y_pred = sigmoid(lr_hyp(x,w))
print(y_pred)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0],x[:,6],y_pred)
plt.show()

#classifying the output data 
y_pred = [1 if p >= y_pred.mean() else 0 for p in y_pred]
print(y_pred)


#finding accuracy,classification error and entropy
accuracy = 0
for i in range(len(y_pred)):
    if y_pred[i] == y[i]:
        accuracy += 1
print(f"Accuracy = {accuracy / len(y_pred)}")
print(f"Cross validation error = {(len(y_pred)-accuracy)/len(y_pred)}")

loss= lr_cost(w,x,y)
print(loss)

#formatting the prediction
y_pred = np.where(y_pred==0,-1,y_pred)
print(y_pred)

#applying the model to test data

dataset1 = pd.read_csv('1644871518_6284828_cleveland-test.csv')
dataset1.head(145)
x1=dataset1.iloc[:,].values
print(x1)
m1,n1= x1.shape
print('x1[0]={}'.format(x1[0]))
m1,n1= x1.shape
z1=np.ones(m1)
z1=z1.reshape(m1,1)
x1=np.append(z1,x1,axis=1)
print('x1[0]={}'.format(x1[0]))
y_hat = sigmoid(lr_hyp(x1,w))
print(y_hat)
y_hat = [1 if p >= y_hat.mean() else 0 for p in y_hat]
print(y_hat)
y_hat = np.where(y_hat==0,-1,y_hat)
print(y_hat)

loss= lr_cost(w,x1,y_hat)
print(loss)
df = pd.DataFrame(y_hat)
df.to_csv('test_result.csv',index=False, header=False)

#checking by normalizing the data
x = normalize(x)
w = grad_descent(x,y,w,eta,iterations)