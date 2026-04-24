import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = {
    'x': [2, 3, 8, 9, 1],
    'y': [4, 5, 7, 8, 2]
}
df = pd.DataFrame(data)
df.to_csv('kmeans_results.csv', index=False)
X = df[['x', 'y']]

initial_centroids = np.array([[2, 4], [8, 7]])
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

print("Final Clusters and Data Points:")
print(df)
print("\nFinal Centroids:")
print(kmeans.cluster_centers_)

plt.scatter(df['x'], df['y'], c=df['Cluster'], cmap='viridis', s=100, label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Final Centroids')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering (K=2)')
plt.legend()
plt.grid(True)
plt.show()