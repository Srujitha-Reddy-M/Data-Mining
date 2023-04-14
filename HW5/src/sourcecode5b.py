#impoting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
image = pd.read_csv("image_new_test.txt", header=None,sep=",") 
print(image.head(10000)) 
X=image.iloc[:,:].values
X
X.shape

#data reduction technique
from sklearn.manifold import TSNE
x = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(X)
x.shape
x

#initializing the centroids to random data points initially
def initialize_centroid(x,k):
  r_index=[np.random.randint(len(x)) for i  in range(k)]
  centroids=[]
  for i in r_index:
    centroids.append(x[i])
  return centroids

#calculating SSE and assigning to respective clustering
def clustering(x,p,k):
  cluster=[]
  for i in range(len(x)):
    dist=[]
    for j in range(k):
      dist.append(np.linalg.norm(np.subtract(x[i],p[j])))
    n=np.argmin(dist)
    cluster.append(n)
  return np.asarray(cluster)

#updating the centroid values
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


#function to plot the centroids and data
def display(x,clusters,centroids,init_cent,mark=True,show_Centroid=True,Plots=True):
  colors={0:'r',1:'b',2:'g',3:'c',4:'m',5:'y',6:'k',7:'orange',8:'0.5',9:'#7b80c7'}
  fig,ax=plt.subplots(figsize=(7.5,6))
  for i in range(len(clusters)):
    ax.scatter(x[i][0],x[i][1],color=colors[clusters[i]])
  for j in range(len(centroids)):
    ax.scatter(centroids[j][0],centroids[j][1],marker='*',color='#00ffff')
    if show_Centroid==True:
      ax.scatter(init_cent[j][0],init_cent[j][1],marker="+",s=150,color='#00ff00')
  ax.set_xlabel("Feature1")
  ax.set_ylabel("Feature2")
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
    print("Final plot(cyan --> final and Green--> initial centroids):")
    display(x,cluster,prev_centroid,init_cent,mark=True,show_Centroid=True)
  return cluster,prev_centroid

cluster,centroid =integration_func(x,10,show ='ini_fi')


print(cluster)

#creating the output file in given format
cluster = np.where(cluster==0,10,cluster)
print(cluster)
df = pd.DataFrame(cluster)
df.to_csv('test2_result.csv',index=False, header=False)

#validating clustering algorithm using the inbuilt library
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10,init = 'k-means++',   max_iter = 100, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
print(kmeans.cluster_centers_)
plt.scatter(x[y_kmeans   == 0, 0], x[y_kmeans == 0, 1],s = 100, c = 'red', label = '1')
plt.scatter(x[y_kmeans   == 1, 0], x[y_kmeans == 1, 1],s = 100, c = 'blue', label = '2')
plt.scatter(x[y_kmeans   == 2, 0], x[y_kmeans == 2, 1],s = 100, c = 'green', label = '3')
plt.scatter(x[y_kmeans   == 3, 0], x[y_kmeans == 3, 1],s = 100, c = 'c', label = '4')
plt.scatter(x[y_kmeans   == 4, 0], x[y_kmeans == 4, 1],s = 100, c = 'm', label = '5')
plt.scatter(x[y_kmeans   == 5, 0], x[y_kmeans == 5, 1],s = 100, c = 'y', label = '6')
plt.scatter(x[y_kmeans   == 6, 0], x[y_kmeans == 6, 1],s = 100, c = 'k', label = '7')
plt.scatter(x[y_kmeans   == 7, 0], x[y_kmeans == 7, 1],s = 100, c = 'orange', label = '8')
plt.scatter(x[y_kmeans   == 8, 0], x[y_kmeans == 8, 1],s = 100, c = '0.5', label = '9')
plt.scatter(x[y_kmeans   == 9, 0], x[y_kmeans == 9, 1],s = 100, c = '#7b80c7', label = '10')

plt.scatter(kmeans.cluster_centers_[:,   0], kmeans.cluster_centers_[:,1],s = 100, c = 'black', label = 'Centroids')
plt.legend()
plt.show()

#Plotting SSE vs no.of clusters (k)
from sklearn.cluster import KMeans
error=[]
for k in range(1,21,2):
  p=KMeans(n_clusters=k, init='k-means++', random_state=25)
  p.fit(x)
  error.append(p.inertia_)
plt.plot(range(1,21,2), error,marker='*')
plt.xlabel('k-value->no.of clusters')
plt.ylabel('error')
plt.show()