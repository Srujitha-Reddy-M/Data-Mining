#impoting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data 
iris = pd.read_csv("iris_new_data.txt", header=None,sep=" ", names=["SepalLength","SepalWidth","PetalLength","PetalWidth"]) 
print(iris.head(150))

#plot the data
iris.plot(kind="scatter", x="SepalLength",   y="SepalWidth")
plt.show()
iris.plot(kind="scatter", x="PetalLength",   y="PetalWidth")
plt.show()

#data preprocessing
x= iris.iloc[:,:].values
x

#function to initialize the centroid to random data points initially
def initialize_centroid(x,k):
  r_index=[np.random.randint(len(x)) for i  in range(k)]
  centroids=[]
  for i in r_index:
    centroids.append(x[i])
  return centroids

#function to calculate SSE and assign data points to respective clusters
def clustering(x,p,k):
  cluster=[]
  for i in range(len(x)):
    dist=[]
    for j in range(k):
      dist.append(np.linalg.norm(np.subtract(x[i],p[j])))
    n=np.argmin(dist)
    cluster.append(n)
  return np.asarray(cluster)


#function to update the centroid value
def calcNewCentroid(x,clusters,k):
  centroid=[]
  for i in range(k):
    arr=[]
    for j in range(len(x)):
      if clusters[j]==i:
        arr.append(x[j])
    centroid.append(np.mean(arr,axis=0))
  return np.asarray(centroid)

#function to calculate the difference between previous centroid position and updated value
#This can be used as threshold as to when to stop the iteration/updation of centroid value
def centroid_diff(prev,next):
  d=0
  for i in range(len(prev)):
    d+=np.linalg.norm(prev[i]-next[i])
  return d

#function to display the necessary plots
def display(x,clusters,centroids,init_cent,mark=True,show_Centroid=True,Plots=True):
  colors={0:'r',1:'b',2:'g'}
  fig,ax=plt.subplots(figsize=(7.5,6))
  for i in range(len(clusters)):
    ax.scatter(x[i][2],x[i][3],color=colors[clusters[i]])
  for j in range(len(centroids)):
    ax.scatter(centroids[j][2],centroids[j][3],marker='*',color=colors[j])
    if show_Centroid==True:
      ax.scatter(init_cent[j][2],init_cent[j][3],marker="+",s=150,color=colors[j])
  if mark==True:
    for i in range(len(centroids)):
      ax.add_artist(plt.Circle((centroids[i][2],centroids[i][3]),0.2,linewidth=3,fill=False,color='c'))
      if show_Centroid==True:
        ax.add_artist(plt.Circle((init_cent[i][2],init_cent[i][3]),0.1,linewidth=3,fill=False,color='m'))
  ax.set_xlabel("PetalLength")
  ax.set_ylabel("PetalWidth")
  ax.set_title("k-means clustering")
  if Plots==True:
    plt.show()


def integration_func(x,k,show='all',Plots=True):
  prev_centroid=initialize_centroid(x,k)
  cluster=clustering(x,prev_centroid,k)
  diff =100
  init_cent = prev_centroid

  if Plots:
    display(x,cluster,prev_centroid,init_cent,Plots=Plots)
  while diff>0.00001:
    cluster= clustering(x,prev_centroid,k)
    if show=='all' and Plots:
      display(x,cluster,prev_centroid,init_cent,False,False,Plots=Plots)
      mark=False
      show_Centroid=False
    new_centroid = calcNewCentroid(x,cluster,k)
    diff = centroid_diff(prev_centroid,new_centroid)
    prev_centroid = new_centroid

  if Plots:
    print("Initial centroids:")
    print(init_cent)
    print("\n")
    print("Final centroids:")
    print(prev_centroid)
    print("\n")
    print("Final plot(cyan indicates final and magnetta indicates initial centroids):")
    display(x,cluster,prev_centroid,init_cent,mark=True,show_Centroid=True)
  return cluster,prev_centroid

cluster,centroid =integration_func(x,3,show ='ini_fi')

#creating the output file in given format
print(cluster)
cluster = np.where(cluster==0,3,cluster)
print(cluster)
df = pd.DataFrame(cluster)
df.to_csv('test1_result.csv',index=False, header=False)




#validating the output using inbuilt function
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init = 'k-means++',   max_iter = 100, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(iris)
print(kmeans.cluster_centers_)
plt.scatter(x[y_kmeans   == 0, 2], x[y_kmeans == 0, 3],s = 100, c = 'red', label = '1')
plt.scatter(x[y_kmeans   == 1, 2], x[y_kmeans == 1, 3],s = 100, c = 'blue', label = '2')
plt.scatter(x[y_kmeans   == 2, 2], x[y_kmeans == 2, 3],s = 100, c = 'green', label = '3')   #Visualising the clusters - On the first two columns
plt.scatter(kmeans.cluster_centers_[:,   2], kmeans.cluster_centers_[:,3],s = 100, c = 'black', label = 'Centroids')   #plotting the centroids of the clusters
plt.legend()
plt.show()

