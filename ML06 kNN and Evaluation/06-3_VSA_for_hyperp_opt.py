"""
06-3_VSA_for_hyperp_opt.py

In this script, we will perform hyperparameter optimization using VSA for the unweighted k-NN model.
    - We will repeat the initialize-fit-predict-evaluate steps for different values of k.
    - We will then find the k with the highest test recognition rate (prediction performance).

To measure performance, we use the Test Recognition Rate (test accuracy).
"""


################ Preliminaries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


################ Load the data
diabetes = pd.read_csv('./Data/diabetes.csv')       # Use this line if you run the code cell by cell.
# diabetes = pd.read_csv('../Data/diabetes.csv')    # Use this line if you run the whole module using the Run button
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array


################ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23, stratify = y)


################ Learn the UNWEIGHTED k-NN model
knn_uw = KNeighborsClassifier(n_neighbors=9)        # 1. Initialize
knn_model_uw = knn_uw.fit(X_train, y_train)         # 2. Fit
y_pred_uw = knn_model_uw.predict(X_test)            # 3. Predict
R_uw = knn_model_uw.score(X_test, y_test)           # 4. Evaluate
print(f"Test Recognition Rate for k={knn_model_uw.n_neighbors} (unweighted): {R_uw * 100:.1f}%")


################ 06 EXERCISE 3: HYPERPARAMETER-OPTIMIZATION for the unweighted k-NN using VSA
#     Repeat the above steps (initialize-fit-predict-evaluate) for the following values of k:
#     k=1       ->  R=
#     k=5       ->  R=
#     k=13      ->  R=
#     k=61      ->  R=
#     k=121     ->  R=
#     k=201     ->  R=
#     k=308     ->  R=
#     For each of them, note the recognition rate in the empty slots above!
#     Based on your results: Which value of k is the best one?
#
#     If you want, your can draw a little graph on paper.
#     Alternatively, you can also aks the AI to implement a loop for you and draw the corresponding plot ;-).

