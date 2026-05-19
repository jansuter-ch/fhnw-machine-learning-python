"""
07-4_CV_ApplicationCase2.py

In this script, we will illustrate Application Case 2 of Cross-Validation (CV):
Model Selection w/o HPO using CV.
    - We compare the 9-Nearest Neighbors Classifier (k-NN) with a Classification Tree with max_depth=5 using CV.
    - To measure prediction performance, we use the Recognition Rate (accuracy).
    - The data set is the diabetes data set.
"""

################ Preliminaries

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold  # For cross-validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

################ Load the data
diabetes = pd.read_csv('./Data/diabetes.csv')

################ Prepare the data
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array

################ Cross-Validation of 9-NN on the entire data set
knn = KNeighborsClassifier(n_neighbors=9)  # Initialize
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold CV on the entire data set
mean_score_knn = np.mean(scores)  # calculate mean accuracy across folds
print(f"Cross-Validation Recognition Rate for k=9: {mean_score_knn * 100:.1f}%")

################ Cross-Validation of Classification Tree with max_depth=5 on the entire data set
dtree = DecisionTreeClassifier(max_depth=5, random_state=23)  # Initialize
scores = cross_val_score(dtree, X, y, cv=5, scoring='accuracy')  # 5-fold CV on the entire data set
mean_score_dtree = np.mean(scores)  # calculate mean accuracy across folds
print(f"Cross-Validation Recognition Rate for Classification Tree with max_depth=5: {mean_score_dtree * 100:.1f}%")

################ Compare the two models based on their CV recognition rates
if mean_score_dtree > mean_score_knn:
    print("The Classification Tree with max_depth=5 performs better than the 9-NN model.")
else:
    print("The 9-NN model performs better than the Classification Tree with max_depth=5.")

################ Train the best model on the entire data set
dtree_model = dtree.fit(X, y)
print("The best model has been trained on the entire data set.")

# REMEMBER:
# - A classification tree is an eager learner (model-based learner).
# - This means that the training step (the "fit" step) involves building the tree structure based on the data.
# - The tree structure is sufficient to make predictions on new data points using the predict method.
#   The training data and the DecisionTreeClassifier algorithm is not needed any more after the tree has been built.


################ Visualize the best Classification Tree model (just for illustration)
fig = plt.figure(figsize=(64,48), dpi=100)
tree.plot_tree(dtree_model,
          feature_names=X.columns,
          class_names=[str(cls) for cls in np.unique(y)],
          rounded=True,
          filled=True,
          proportion=True,
          fontsize=10);
plt.show()

# REMARK:
# - A Decision Tree is a WHITE-BOX MODEL, meaning its decisions can be easily interpreted.
# - The visualization above shows the structure of the tree, including the features used for splits,
#   the thresholds for those splits, and the class distributions at the leaf nodes.
# - This transparency makes Decision Trees particularly useful in applications where INTERPRETABILITY is crucial.
#   It can be used to explain the model's decisions to stakeholders who may not have a technical background.
