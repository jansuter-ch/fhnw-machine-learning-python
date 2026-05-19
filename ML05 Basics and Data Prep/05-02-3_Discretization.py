"""

05-02-3_Discretization.py
Transforming numerical to categorical.

"""

import pandas as pd
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt


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

data.info()
data.head()
num_data = data.select_dtypes(include=['int64', 'float64'])
num_data.info()
num_data.head()


##### We selet only the first 2 numerical features
num_data_12 = num_data.iloc[:, :2]

# We convert the DataFrame to a numpy ndarray using the method .to_numpy() from pandas.
#   - This is necessary, because the KBinsDiscretizer() from sklearn only works with ndarrays.
num_data_12_array = num_data_12.to_numpy()
type(num_data_12_array) # numpy.ndarray
num_data_12_array.dtype # underlying data type: dtype('int64')
num_data_12_array.shape # (32561, 2) - 2 numerical features

# Let's plot a histogram of all features:
num_data_12.hist(bins=30, figsize=(10,8))
plt.tight_layout()
plt.show()

##### Equal Width Binning

# We use pre.KBinsDiscretizer()
#
#   NOTE: For the "initialize-fit-transform" process in scikit-learn, check the info-sheet on Moodle!
#
#   1. Initialize
#       Create an instance of the KBinsDiscretizer() class by specifying the parameters:
#       - n_bins = 3 (we want to have 3 intervals/bin)
#       - strategy = 'uniform' (we want to use Equal Width Binning)
#       - encode = 'ordinal' (the interval identifiers are encoded as integer values)
ewb = pre.KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
#
#   2. Fit
#       Calculate the bin edges based on the specified number of bins (3) and the strategy used ('uniform').
#       The bin edges are stored in the attribute 'bin_edges_' of 'ewb'.
ewb.fit(num_data_12_array)
print(ewb.bin_edges_)
#       - Remember that we have 2 features in num_data_array.
#       - ewb.fit() was applied to each of them separately.
#       - For each of the 2 features, we get 4 bin edges (thus 3 bins)
#
#   3. Transform
#       Maps the original numerical values to the corresponding bins.
#       Remember that the bin identifiers are encoded as floats 0., 1., 2. (because we set encode='ordinal')
num_data_12_array_ewb = ewb.transform(num_data_12_array)
num_data_12_array.shape # (32561, 2)

# IMPORTANT NOTE:
#   - The KBinsDiscretizer only *discretizes* the numerical values,
#   - it does not actually make them categorical!
#   - If we want to make sure that an ML algorithm infers a categorical scale
#   - after discretization, we must convert the bin identifiers to string:
num_data_12_array_ewb_cat = num_data_12_array_ewb.astype(str)

# Now we convert the ndarray back to DataFrame using the method .DataFrame() from pandas:
num_data_12_ewb_cat = pd.DataFrame(num_data_12_array_ewb_cat, columns=num_data_12.columns)
num_data_12_ewb_cat.info() # All features are categorical ('object')

# Let's plot a bar chart of all features and compare it with the original data:
fig, axes = plt.subplots(1, len(num_data_12_ewb_cat.columns), figsize=(12, 5))
for i, col in enumerate(num_data_12_ewb_cat.columns):
    num_data_12_ewb_cat[col].value_counts().sort_index().plot.bar(ax=axes[i])
    axes[i].set_title(f'ewb Balkendiagramm für {col}')
    axes[i].set_xlabel('Kategorie')
    axes[i].set_ylabel('Anzahl')
plt.tight_layout()
plt.show()

# Print the value counts for each feature:
for col in num_data_12_ewb_cat.columns:
    print(num_data_12_ewb_cat[col].value_counts())



##### Equal Frequency Binning
#   To use it, we must again follow the "initialize-fit-transform" process.
#
#   1. Initialize
#       - strategy = 'quantile' (we want Equal Frequency Binning now)
efb = pre.KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')

#   2. Fit
efb.fit(num_data_12_array)
print(efb.bin_edges_)

#   3. Transform
num_data_12_array_efb = efb.transform(num_data_12_array)
num_data_12_array_efb.shape # (32561, 2)

# Make categorical by converting to string:
num_data_12_array_efb_cat = num_data_12_array_efb.astype(str)

# Transform back to data frame  :
num_data_12_efb_cat = pd.DataFrame(num_data_12_array_efb_cat, columns=num_data_12.columns)
num_data_12_efb_cat.info() # All features are categorical ('object')

# Plot a bar chart of all features and compare it with the original data:
fig, axes = plt.subplots(1, len(num_data_12_efb_cat.columns), figsize=(12, 5))
for i, col in enumerate(num_data_12_efb_cat.columns):
    num_data_12_efb_cat[col].value_counts().sort_index().plot.bar(ax=axes[i])
    axes[i].set_title(f'efb Balkendiagramm für {col}')
    axes[i].set_xlabel('Kategorie')
    axes[i].set_ylabel('Anzahl')
plt.tight_layout()
plt.show()

# Print the value counts for each feature:
for col in num_data_12_efb_cat.columns:
    print(num_data_12_efb_cat[col].value_counts())