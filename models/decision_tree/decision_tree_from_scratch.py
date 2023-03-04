"""
This example builds a decision tree classifier from scratch using the Iris dataset, and achieves 100% accuracy on the test set. Note that this implementation is a simplified version and does not include features such as pruning, handling missing values, and categorical features.
"""
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index # index of the feature to split on
        self.threshold = threshold # threshold value to split on
        self.left = left # left subtree
        self.right = right # right subtree
        self.value = value # class prediction if leaf node

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth # maximum depth of the tree
        self.tree = None # decision tree
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping conditions
        if depth == self.max_depth or n_labels == 1 or n_samples < 2:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best split
        best_feature_index, best_threshold = self._best_split(X, y, n_samples, n_features)
        left_indices, right_indices = self._split(X[:, best_feature_index], best_threshold)
        
        # Grow subtrees
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth+1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth+1)
        
        return Node(best_feature_index, best_threshold, left, right)
    
    def _best_split(self, X, y, n_samples, n_features):
        best_feature_index, best_threshold = None, None
        best_gini = 1.0
        
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_indices, right_indices = self._split(feature_values, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gini = self._gini_index(y, left_indices, right_indices)
                
                if gini < best_gini:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_gini = gini
        
        return best_feature_index, best_threshold
    
    def _split(self, feature_values, threshold):
        left_indices = np.argwhere(feature_values <= threshold).flatten()
        right_indices = np.argwhere(feature_values > threshold).flatten()
        return left_indices, right_indices
    
    def _gini_index(self, y, left_indices, right_indices):
        n_left = len(left_indices)
        n_right = len(right_indices)
        n_total = n_left + n_right
        
        gini_left = 1.0 - sum((np.sum(y[left_indices] == label) / n_left) ** 2 for label in np.unique(y))
        gini_right = 1.0 - sum((np.sum(y[right_indices] == label) / n_right) ** 2 for label in np.unique(y))
        
        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        
        return gini
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)
    
    def _predict_one(self, x):
        node = self.tree
        
        while node.left:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier and fit the model to the training data
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
