# Decision Tree

## Introduction
A decision tree is a non-parametric supervised learning algorithm used for classification and regression tasks. It is a type of model that can be used to predict the value of a target variable based on several input variables, by splitting the data into subsets based on simple rules inferred from the input features.

The decision tree is constructed by recursively splitting the data into smaller subsets, based on the features that lead to the highest information gain or Gini impurity reduction. At each split, the algorithm chooses the feature and the threshold that maximally separates the data into classes with the highest purity. The final result is a tree-like structure that contains a set of rules for classification or prediction.

Decision trees have the advantage of being easy to interpret and visualize, as the resulting tree can be easily displayed graphically. They can also handle both numerical and categorical input variables, and are less sensitive to outliers than other models. However, they can be prone to overfitting, and may not perform as well as other models in some cases.


### Entrop vs Gini Impurity
Entropy and Gini impurity are two commonly used measures of impurity or randomness in a dataset used in decision tree algorithms. While they are mathematically different, they both serve the same purpose of helping to determine the optimal split of the dataset at each node of the decision tree.

The main difference between entropy and Gini impurity lies in the way they measure the degree of impurity. Entropy measures the degree of disorder or randomness in a dataset, while Gini impurity measures the probability of incorrectly classifying a randomly chosen element in the dataset according to the distribution of classes in the dataset.

The formula for entropy is as follows:

Entropy = -∑(pi)log2(pi)

where pi is the probability of an element in the dataset belonging to a particular class.

The formula for Gini impurity is as follows:

Gini impurity = 1 - ∑(pi)^2

where pi is the probability of an element in the dataset belonging to a particular class.

Both measures have their own advantages and disadvantages. In general, entropy tends to create more balanced splits, while Gini impurity tends to be faster to compute. However, the difference in performance between the two measures is usually minimal, and the choice of measure ultimately depends on the specific characteristics of the dataset and the decision tree algorithm being used.

In practice, most decision tree algorithms provide an option to use either entropy or Gini impurity as the splitting criterion, and it is often recommended to try both measures and choose the one that produces the best results on the specific dataset being used.

## Information Gain
Information gain is a measure of the amount of information provided by a feature in a dataset used in decision tree algorithms. It is used to determine the optimal split of the dataset at each node of the decision tree by selecting the feature that provides the most information about the class labels.

The basic idea behind information gain is to measure the reduction in entropy or Gini impurity achieved by splitting the dataset on a particular feature. The feature that achieves the highest reduction in entropy or Gini impurity is selected as the splitting criterion at that node.

The formula for information gain based on entropy is as follows:

Information Gain = Entropy(S) - ∑((|Sv|/|S|)*Entropy(Sv))

where S is the original dataset, Sv is the subset of S for which the value of the feature being considered is equal to v, and |Sv| and |S| are the number of elements in Sv and S, respectively.

The formula for information gain based on Gini impurity is similar:

Information Gain = Gini(S) - ∑((|Sv|/|S|)*Gini(Sv))

where Gini(S) is the Gini impurity of the original dataset, and Gini(Sv) is the Gini impurity of the subset Sv.

In general, features with higher information gain are preferred, as they provide more information about the class labels and lead to more accurate and efficient decision trees. However, the choice of splitting criterion ultimately depends on the specific characteristics of the dataset and the decision tree algorithm being used.

The formula for information gain using entropy is:

Information Gain = Entropy(S) - ∑((|Sv|/|S|)*Entropy(Sv))

where S is the original dataset, Sv is the subset of S for which the value of the feature being considered is equal to v, and |Sv| and |S| are the number of elements in Sv and S, respectively.

This formula is based on the concept of entropy, which measures the degree of disorder or randomness in a dataset. The idea behind using entropy to calculate information gain is to select the feature that maximally reduces the entropy of the dataset, as this feature will provide the most information about the class labels.

In practice, the choice of information gain based on entropy versus information gain based on Gini impurity often depends on the specific characteristics of the dataset and the decision tree algorithm being used. In general, information gain based on entropy tends to be more sensitive to the distribution of classes in the dataset, while information gain based on Gini impurity tends to be faster to compute. It is often recommended to try both measures and choose the one that produces the best results on the specific dataset being used.





