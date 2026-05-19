"""

05-02-1_Data_Exploration.py
Gaining Data Understanding by exploring the data set.

"""

import pandas as pd
import matplotlib.pyplot as plt


##### Import the census data set

#  Note: Please download 'census.data'.

# We import the census data set using pd.read_csv()
#   - pd.read_csv() imports a .csv file as a data frame
#   - 'census.data' does not have a header that contains the column names. Therefore we set header=None.
#   - To specify a header (the column names) manually, we use names = ... .
# Note:
#   - If you run your code cell by cell, your working directory is your project file.
#     In this case, use the relative path ./Data/census.data.
#   - If you the whole module using the Run button, your working directory is the current file.
#     In this case use the relative path ../Data/census.data.

data = pd.read_csv('./Data/census.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                            'education', 'education-num', 'marital- status', 'occupation',
                            'relationship', 'race', 'gender', 'capital-gain',
                            'capital-loss', 'hours-per-week', 'native-country', 'income'])

# data = pd.read_csv('../Data/census.data', header=None, index_col=False,
#                    names=['age', 'workclass', 'fnlwgt',
#                             'education', 'education-num', 'marital- status', 'occupation',
#                             'relationship', 'race', 'gender', 'capital-gain',
#                             'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Check if it is really a data frame
type(data)


##### A little bit of data exploration

# If you work with a data set for a customer, first thing to do is business understanding and data understanding.
#   - Business understanding: What is the goal of the analysis? What do you want to achieve?
#   - Data understanding: What data do you have? What is the meaning of the variables? What are the data types and scale levels?
#                         Are there missing values? Are there outliers? What are the summary statistics?
#   - Data understanding is an iterative process. You might need to go back and forth between business understanding and data understanding.
#     E.g., you might need to ask the customer what the meaning of a variable is.
#   - Data understanding is a crucial step in the data science process. If you don't understand the data, you cannot make sense of the analysis results.
#     Garbage in, garbage out!


# First, let's have a look at our data frame
print(data.head()) # We don't see all columns :-/
pd.set_option('display.max_columns', None) # To see all columns of a data frame, change default view with pd.set_option()
print(data.head()) # Now we see all columns :-)


# Display the structure of the data frame
data.info()
#   What do we see here?
#   - We have 15 variables and 32561 observations.
#   - The data types are either 'object' or 'int64'.
#       - 'object' means categorical.
#       - int64 means numerical
#   - Note: In Pandas, 'object' is a catch-all for any data type that doesn't fit into the standard numeric types.
#           Thus, it can also contain strings or mixed types.
#   - We see that there are no missing values (non-null = 32561 for all variables).


# Now let's display the summary statistics of the columns
data.describe()
#   Notice that only the numerical variables are displayed!
#   The reason is that 'describe()' summarizes only numerical variables by default.
#   To include a summary of the categorical variables, you must set the parameter include='all':
data.describe(include='all')
#   Now we also see the categorical variables.
#   - 'count' shows the number of non-missing values of a variable.
#   - 'unique' shows the number of unique values of a categorical variable. E.g., workclass has 9 unique values.
#     To display the unique values of a categorical variable, use unique():
data['workclass'].unique()
data['education'].unique()
daat['marital- status'].unique()
data['occupation'].unique()
data['relationship'].unique()
data['race'].unique()
data['gender'].unique()
data['native-country'].unique()
data['income'].unique()
#   - 'top' shows the most frequent value of a categorical variable.
#   - 'freq' shows the frequency of the most frequent value of a categorical variable.
#   - 'mean', 'std', 'min', '25%', '50%', '75%', and 'max' show the respective summary statistics of a numerical variable.

# The summary statistics already gives you a lot of information for data understanding. E.g. age:
data['age'].describe()
#   - E.g., the data set contains 32'561 people.
#   - E.g., the youngest person is 17 years old, the oldest person is 90 years old.
#   - E.g., the average age is 38.6 years.
#   - E.g., the median age is 37 years. This means that half of the people are younger than 37 years, half are older.
#   - E.g., age is thus slightly right-skewed (mean > median). This is because there are a bit more younger people than older people.
data['hours-per-week'].describe()
#   - E.g., 50% of the people work at most 40 hours per week.
#   - E.g., the maximum number of hours worked per week is 99 hours. This is probably an outlier!
data['capital-gain'].describe()
data['capital-loss'].describe()
#   - E.g., capital-gain and capital-loss are heavily right-skewed (mean >> median).
#     This is because most people have no capital gain or loss at all: median = 0!
#     but a few people have very high capital gain or loss (max = 99999).
#   - Let's plot the histograms to see the distribution of capital-gain:
data['capital-gain'].hist(bins=50)
plt.xlabel('Capital Gain')
plt.ylabel('Frequency')
plt.title('Histogram of Capital Gain')
plt.show()  # This will display the plot
#   - Plot the histograms to see the distribution of capital-loss:
data['capital-loss'].hist(bins=50)
plt.xlabel('Capital Gain')
plt.ylabel('Frequency')
plt.title('Histogram of Capital Gain')
plt.show()  # This will display the plot
#   - The variables 'capital-gain' and 'capital-loss' can still be useful for analysis.
#   - Maybe most people have no capital gain or loss at all, but the few who have it might be important for predicting income.

data['workclass'].describe()
#   - E.g., most people (24'279) work in the private sector.

data['education'].describe()
#   - E.g., most people (10'099) have a HS-grad education
# Let's plot the distribution of education:
data['education'].value_counts().plot(kind='bar')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.title('Distribution of Education Level')
plt.show()  # This will display the plot

# Visual data exploration is a big topic on its own.
#   There is much more you can do.
#   But this is beyond the scope of this course.
#   If you are interested, check out:
#       - Book "Hands-On Exploratory Data Analysis with Python": https://www.oreilly.com/library/view/hands-on-exploratory-data/9781789537253/
#       - The Pandas documentation: https://pandas.pydata.org/docs/user_guide/visualization.html
#       - The Seaborn documentation: https://seaborn.pydata.org/tutorial.html
#       - The Matplotlib documentation: https://matplotlib.org/stable/tutorials/index.html

