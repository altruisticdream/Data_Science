#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[28]:


dataset=pd.read_csv("/Users/ashutoshgautam/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv")
X=dataset.iloc[:, [3,4]].values
dataset.head(5)


# In[29]:


#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++",random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()


# In[30]:


kmeans=KMeans(n_clusters=5, init="k-means++",random_state=42)
y_kmeans=kmeans.fit_predict(X)
print(y_kmeans)


# In[31]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s=100 ,c='red' ,label='cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=100 ,c='blue' ,label='cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=100 ,c='green' ,label='cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s=100 ,c='cyan' ,label='cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s=100 ,c='magenta' ,label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title("Cluster of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-1000)")
plt.show()

