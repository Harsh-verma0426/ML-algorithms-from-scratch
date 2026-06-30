import numpy as np

class KNeighborsRegressor:
  def __init__(self, n_neighbors):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
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
      avg = np.mean([target for _, target in neighbors])
      final.append(avg)

    return np.array(final)