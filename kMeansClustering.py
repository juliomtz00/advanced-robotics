import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate two separate datasets with 20 (x, y) points each
np.random.seed(0)

# Dataset 1
data1 = np.random.rand(20, 2) * 10

# Dataset 2
data2 = np.random.rand(20, 2) * 10 + 20

# Combine both datasets
data = np.vstack((data1, data2))

# Apply K-means clustering with K=2 (since we want two center points)
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Get the final center points
centers = kmeans.cluster_centers_

# Separate the data into two sets based on cluster assignments
labels = kmeans.labels_
data1 = data[labels == 0]
data2 = data[labels == 1]

# Plot the datasets and centers
plt.figure(figsize=(8, 6))

# Plot Dataset 1 in blue
plt.scatter(data1[:, 0], data1[:, 1], c='blue', label='Dataset 1')

# Plot Dataset 2 in red
plt.scatter(data2[:, 0], data2[:, 1], c='red', label='Dataset 2')

# Plot the centers as green stars
plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='*', s=200, label='Centers')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('K-Means Clustering with Centers')
plt.legend()
plt.grid(True)
plt.show()
