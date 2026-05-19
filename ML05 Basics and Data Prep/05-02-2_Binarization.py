"""

05-02-2_Binarization.py
Also called One-Hot-Encoding or Dummy Variable Approach.
Transforming categorical to numerical.

"""
from xml.etree.ElementInclude import include

import pandas as pd


##### Import the census data set again

data = pd.read_csv('./Data/census.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                            'education', 'education-num', 'marital- status', 'occupation',
                            'relationship', 'race', 'gender', 'capital-gain',
                            'capital-loss', 'hours-per-week', 'native-country', 'income'])

# If you the whole module using the Run button, your working directory is the current file.
# In this case use the relative path ../Data/census.data:
# data = pd.read_csv('../Data/census.data', header=None, index_col=False,
#                    names=['age', 'workclass', 'fnlwgt',
#                             'education', 'education-num', 'marital- status', 'occupation',
#                             'relationship', 'race', 'gender', 'capital-gain',
#                             'capital-loss', 'hours-per-week', 'native-country', 'income'])


#####  Select a subset of 4 variables and inspect them
my_data = data[['age', 'workclass', 'gender', 'income']]
print(my_data.head())
my_data.info() # Check the data types
my_data.describe(include='all')
my_data['workclass'].unique()
my_data['gender'].unique()
my_data['income'].unique()


##### Binarize the 3 categorical variables
#
#   Remember: Binarization (One-Hot-Encoding, Dummy Variable Approach)
#             transforms categorical variables to numerical variables.

#   We use the pandas function 'get_dummies()' to do that.
my_data_num = pd.get_dummies(my_data)

# Inspect what happened:
my_data_num.info()
    # We see that we have many more variables now:
        # age: is untouched, because it was already numerical
        # workclass: 9 dummies
        # gender: 2 dummies
        # income: 2 dummies
    # All dummies are boolean.

# Compare the dummies with the unique values of the categorical features before transformation:
my_data['workclass'].unique()
my_data['gender'].unique()
my_data['income'].unique()
# We see that there is one boolean dummy per unique value of the categorical features.
pd.set_option('display.max_columns', None) # To see all columns of a data frame
my_data_num.head()