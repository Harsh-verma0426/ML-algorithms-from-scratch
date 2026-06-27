import numpy as np
from collections import Counter

class Node:
  def __init__(self,
              feature = None,
              threshold = None,
              left = None,
              right = None,
              value = None,
              gain = None
          ):
    
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
    self.gain = gain

def entropy(y):
  classes, counts = np.unique(y, return_counts=True)
  probabilities = counts/len(y)
  entropy = -np.sum(probabilities * np.log2(probabilities))
  return entropy

def information_gain(y, left_idx, right_idx, parent_entropy):
  y_left = y[left_idx]
  y_right = y[right_idx]
  if len(y_left) == 0 or len(y_right) == 0:
    return 0
  entropy_left = entropy(y_left)
  entropy_right = entropy(y_right)
  weight_left = len(y_left)/len(y)
  weight_right = len(y_right)/len(y)
  child_entropy = (entropy_left*weight_left) + (entropy_right*weight_right)
  return parent_entropy-child_entropy

def split(X, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return left_idx, right_idx

def find_best_split(X, y):

  best_gain = -1
  parent_entropy = entropy(y)
  for feature in range(X.shape[1]):
    
    thresholds = np.unique(X[:, feature])

    for threshold in thresholds:
      left_idx, right_idx = split(X, feature, threshold)
      
      gain = information_gain(y, left_idx, right_idx, parent_entropy)
      if gain>best_gain:
        best_gain = gain
        best_node = Node(
              feature=feature,
              threshold=threshold,
              gain=gain
          )

  return best_node

def most_common_label(y):
  mode = Counter(y).most_common(1)[0][0]
  return mode

def build_tree(X, y):
  if len(np.unique(y)) == 1:
    return Node(value=y[0])
  node = find_best_split(X, y)
  if node.gain <= 0:
    return Node(value=most_common_label(y))
  left_idx, right_idx = split(X, node.feature, node.threshold)
  node.left = build_tree(X[left_idx], y[left_idx])
  node.right = build_tree(X[right_idx], y[right_idx])
  return node

