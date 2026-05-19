"""
06-1_kNN.py
In this module, we apply a k-NN classifier and a weighted k-NN Classifiers to the diabetes dataset.
    - First, we learn an unweighted k-NN model from the diabetes dataset.
    - Second, we learn a weighted k-NN model from the diabetes dataset.
We use both models to predict whether a new patient has diabetes or not.
"""

################ Preliminaries

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


################ Load the data

#       We import the diabetes dataset (see github). It originally stems from Kaggle.
#       It holds data on diabetes tests. Our goal is to predict the column 'Outcome'.
#       Since 'diabetes' is a csv file, we can use read_csv() from pandas to load it as a data frame.
diabetes = pd.read_csv('./Data/diabetes.csv') # Use this line if you run the code cell by cell.
# diabetes = pd.read_csv('../Data/diabetes.csv') # Use this line if you run the whole module using the Run button
print(diabetes) # Alternatively, view it in the data Viewer of PyCharm.

#       We want to use the KNeighborsClassifier algorithm from scikit-lkearn for classification.
#       It requires that the input features and the target variable are provided separately.
#       So, let's separate input and target:
X = diabetes.drop(columns=['Outcome']) # Use the pandas method .drop to drop the target column
y = diabetes['Outcome'] # Use the bracket notation [] to select only the target column

#      KNeighborsClassifier additionally needs the target variable to be provided as a numpy array.
y = y.values # Use the pandas method .values to convert the data frame into a numpy array.

#       Double-check the data types of X and y:
type(X) # pandas.core.frame.DataFrame
type(y) # numpy.ndarray



################ "Learn the k-NN model from the sample data"
# (Aka: "Fit the k-NN algorithm to the sample data")

#   To learn a model from the sample data using the kNN algorithm, we apply the initialize-fit-predict process.
#       - The initialize-fit-predict process is used to fit a machine learning model to sample data, and then use it
#         to make predictions on new data.
#       - It is similar to the initialize-fit-transform process used for data preprocessing.
#       - See the Memo posted on Moodle for more details.
#
#   1. Initialize
#      We create an instance of the KNeighborsClassifier() by specifying its parameters.
#      Here, we only need to specify one parameter, namely the number of neighbours:
#       - n_neighbors = 9. The 'n_neighbors' parameters specifies the number of neighbours k that kNN uses
#         for the majority vote. As a first attempt, we choose k=9.
#      Notice that we choose an odd number for k to avoid ties in the majority vote.
knn = KNeighborsClassifier(n_neighbors=9)
#
#   2. Fit
#      We fit the kNN algorithm to the whole sample data (X,y).
#      Note:
#       - Normally, in this step, the algorithm learns a model from the sample data. Afterwards we don't need the
#         algorith any morem and we also don't need the sample data any more.
#         The model is all we need to make predictions on new data.
#         E.g., the DecisionTreeClassifier() algorithm learns the *tree structure* from the sample data.
#         The tree structure is the model. Once we have the tree structure, we can use it to make predictions.
#         We don't need the sample data for that, and we also don't need the learning algorithm any more.
#       - This is different for kNN, because kNN is a lazy learner!
#         This means that we cannot separate the model from the sample data and from the learning algorithm.
#         Let's see what that means:
knn_model = knn.fit(X, y) # We take the instance of the kNN algorithm from step 1, and "learn the model".
#       - Yet, actually, not much happens in this step.
#         Since kNN is a lazy learner, this step only stores the sample data and the decision rule.
#       - Thus, when we look into the model object 'knn_model', we see that the data and the algorithm are stored in it:
#           - '_fit_X' is an attribute of the model object 'knn_model'. It stores the input features of the data sample:
print(knn_model._fit_X)
#           - '_y          ' is an attribute of the model object 'knn_model'. It stores the target values of the data sample:
print(knn_model._y)
#           - 'n_neighbors' is an attribute of the model object 'knn_model'. It stores the number of neighbours used to
#                           make the majority vote for prediction. This is the parameter we chose in the initialization step:
print(knn_model.n_neighbors)
#   - Thus, when we "learn a model" with the kNN algorithm, we actually don't learn anything. We only store the sample data
#     and the decision rule in an object that we call "the model object".

#   3. Predict
#       We now want to "apply the model" to new data.
#       In other words, our customer (the hospital) has a new patient, and we want to predict whether this patient has diabetes or not.
#       The new patient's data is as follows:
#           Pregnancies: 1
#           Glucose: 123
#           BloodPressure: 100
#           SkinThickness: 4
#           Insulin: 135
#           BMI: 39.4
#           DiabetesPedigreeFunction: 0.422
#           Age: 44
new_patient = np.array([[1, 123, 74, 4, 135, 39.4, 0.422, 44]]) # Create a 1D numpy array for the new patient.
new_patient = pd.DataFrame(new_patient, columns=X.columns) # Convert the array to a data frame with the same column names as X.
#       Now, we can use the "model" stored in the model object 'knn_model' to predict the class value of the new patient:
new_patient_pred = knn_model.predict(new_patient) # To do that, we can use the method .predict().
print(f"Predicted class value for the new patient (k={knn_model.n_neighbors}): {new_patient_pred}") # Remark: Using "f-strings" for formatted printing here.
#       The predicted class value is 0, which means that the model predicts that the new patient does not have diabetes.

################  06 EXERCISE 1: UNWEIGHTED k-NN
#   - Repeat the above steps (initialize-fit-predict) for a few different values of k, e.g. k=1, k=3, k=5, k=7, k=9, k=11
#     Remember that you can specify k in the initialization step by setting the parameter n_neighbors.
#   - What do you observe?
#   - Can you decide on this basis which value of k you should choose?




################ "Learn the WEIGHTED k-NN model from the sample data"
# (Aka: "Fit the weighted k-NN algorithm to the sample data")
#
# Remember:
#       - The weighted k-NN algorithm is a variant of the k-NN algorithm.
#       - In weighted k-NN, the neighbours are weighted according to their distance to the new object.
#       - Neighbours that are closer to the new object have a higher weight than neighbours that are farther away.
#       - Thus, the closer neighbours have a higher influence on the prediction than the farther neighbours.


# To learn a model from the sample data using the weighted kNN algorithm, we again apply the initialize-fit-predict process,
# just like above. The only difference is that we need to specify an additional parameter in the initialization step.
#
#
#   1. Initialize
#       We use the same algorithm KNeighborsClassifier(), but with an additional parameter:
#       - n_neighbors = 9.
#       - weights='distance'. (The neighbours are weighed according to their distance to the new object.)
knn_w = KNeighborsClassifier(n_neighbors=9, weights='distance')
#
#   2. Fit
knn_w_model = knn_w.fit(X, y) # Same as above
#
#   3. Predict
new_patient_pred_w = knn_w_model.predict(new_patient) # Same as above
print(f"Predicted class value for the new patient (k={knn_w_model.n_neighbors}, weighted): {new_patient_pred_w}")


################ 06 EXERCISE 2: WEIGHTED k-NN
#   - Repeat the above steps (initialize-fit-predict) for a few different values of k, e.g. k=1, k=3, k=5, k=7, k=9, k=11
#   - What do you observe now? Does the weighting change the predictions compared to the unweighted kNN for this patient?
#   - Can you decide on this basis whether you should use unweighted or weighted k-NN?
