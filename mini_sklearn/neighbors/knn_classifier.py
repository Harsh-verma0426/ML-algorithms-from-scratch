import numpy as np
from collections import Counter

class KNeighborsClassifier:
  def __init__(self, n_neighbors):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
    
    if self.n_neighbors > len(self.X_train):
      raise ValueError("n_neighbors cannot exceed number of training samples")

    return self

  def predict(self, X):
    final = []
    for i in range(len(X)):
      distances = []
      for j in range(len(self.X_train)):
        distance = np.sqrt(np.sum((self.X_train[j]-X[i])**2))
        distances.append((distance, self.y_train[j]))
      
      distances.sort(key=lambda x: x[0])
      neighbors = distances[:self.n_neighbors]
      labels = [label for _, label in neighbors]
      mode = Counter(labels).most_common(1)[0][0]
      final.append(mode)

    return np.array(final)