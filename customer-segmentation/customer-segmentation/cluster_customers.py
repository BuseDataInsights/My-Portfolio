import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
data = pd.read_csv('Mall_Customers.csv')

# Gerekli sütunları seçme
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# K-means modeli oluşturma
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# Grupları ekleme
data['Cluster'] = kmeans.labels_

# Grupları görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set1')
plt.title('Customer Segments')
plt.show()
