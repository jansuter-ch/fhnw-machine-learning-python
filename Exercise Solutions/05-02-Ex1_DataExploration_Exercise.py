"""

05-2-Ex1_Data_Exploration_Exercise.py

"""

import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
data = pd.read_csv('./Data/winequality-red.csv', sep=';')

# 1. Show the first five rows
pd.set_option('display.max_columns', None) # Show all columns
print(data.head())

# 2. Show the structure of the DataFrame
data.info()
#    Remark: You can double-check the number of missing values with
print(data.isnull().sum())

# 3. Show statistical metrics
print(data.describe()) # Only numerical features are present
#   - If there were categorical features, you could use
#     print(data.describe(include='all'))

#    Create histograms for numerical features
data.hist(bins=30, figsize=(12, 10))
plt.tight_layout()
plt.show()

# 4. Create boxplots for volatile acidity, citric acid, residual sugar, free sulfur dioxide, total sulfur dioxide, sulphates
potential_outliers = ['volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
data[potential_outliers].plot(kind='box', subplots=True, layout=(2, 3), figsize=(12, 8))
plt.tight_layout()
plt.show()

# 5. Visualize the two-dimensional distributions of all numerical features using scatter plots
pd.plotting.scatter_matrix(data, figsize=(15, 15))
plt.tight_layout()
plt.show()

# 6. Visualize the distribution of the variable 'quality' as a separate bar chart
data['quality'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality')
plt.show()



