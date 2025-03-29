---
title: "Supervised Learning for Credit Risk Assessment"
excerpt: "A simple, step-by-step assessment of credit risk on a credit dataset. The data exploration uses Pyspark and Pandas, and ML implementation uses Scikit-learn.
"
collection: portfolio
entries_layout: grid
classes: wide
---
<p align="justify">
I have always been fascinated by how the morgage and lending system works. Fundamentally speaking, all human endeavors to progress as a civilization need resources espcially in form of finance for execution. Whether it is government funding for research, starting a new company to provide a solution/service to the population, or to buold the next breakthrough techonology, everything needs financial resources to succeed. 
</p>
<p align="justify">
However, in today's world is complicated, so it the process of investing. Thus, financial institutions, over a pperiod of time developed a methodologies to systematically compute the risk associated with providing any financial support buy studying the patterns for   
</p>
<p align="justify">
Here, I analyze a credit risk dataset to check if I can use supervised learning strategies to predict default chances. I will first load and explore the data using Pyspark. Pyspark also also easy data exploration like pandas. Then we will use scikit-learn to perform some supervised learning. I will mainly try Logistic Regression Classifier, bagged tree classifier using random Forests, K-Nearest Neighbor Search, and Support Vector Machines for predicting the defaulting cases. I have added code blocks to describe the process, but you can find a 'RUN-ready' <a href ='https://colab.research.google.com/drive/1xQtpyV824M2Gl-wsdF8CkGhBHT0KKcxv?usp=sharing'>notebook</a> on Google Colab.
</p>
<p align="justify">
First, we import all the necessary packages 
for data manipulation: pypspark, pandas and numpy, Plotting tools: matplotlib and seaborn, supervised ML: scikit-learn-LogisticRegression,  KNeighborsClassifiers, SVM (linear, poly and rbf). We test the quality of the predictions using these techniques: sklearn-metrics. 
Before applying ML operations, the data has to be preprocessed to make it ready for analysis. For this, we use imputing, column transformation, and pipelining modules of scikit-learn. 
</p>

``` python
  # import some packages here
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#  import Spark session to create a session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkFiles

# import scikit-learn regression modules
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# import metrics and preprocessing tools
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# treating missing data treatment
from sklearn.impute import SimpleImputer
# For different preprocessing for different columns
from sklearn.compose import ColumnTransformer
# Develop preprocessing pipeline
from sklearn.pipeline import Pipeline
```

<p align="justify">
We start of by creating a spark session and then load the data from my Github repo as a Pyspark dataframe.
</p>

```python 
  
  # Start a pyspark session
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .master("local[2]") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
	
	# load the data from the url
url = 'https://github.com/Samadarshi-Maity/Credit_Risk_Assessment/raw/main/credit_risk_dataset.csv'

# unlike pandas we cannot directly read from url but need to import into the cluster
spark.sparkContext.addFile(url)

# read the csv data
data  = spark.read.csv('file://'+SparkFiles.get("credit_risk_dataset.csv"), header=True, inferSchema=True)

# push the data into pandas format for the api to operate
  ```
## Understanding the dataset
<p align='justify'>
This data set is a very simple one: it contains the credit default information of the customers. It contains columns with demographic info like the customer's age and income, and their borrowing  information like intent, status, quantity, default history, etc.  
</p>

<p align="justify">
We start by looking at the general problems of raw datasets, like duplicates and missing values. Depending on the analysis protocols, we can remove the duplicates. For the missing values, we can use several strategies, like filling them with the mean, mode, or median data of the column or leaving them empty altogether. Each of these is implemented in a  specific condition and comes with a set of advantages and disadvantages. E.g., for a population data with a skewed distribution, the 'Mean' value is suited for continuous data but poorly  represents the outliers. On the other hand, 'median' is a better choice for skewed data but is not suitable for continuous data. Finally, we can use more elaborate strategies like KNN or polynomial fits to fill missing values that use other features to guess/predict the most likely value of the missing data, but again, they have their respective flaws.
</p>

<p align='justify'>
The code block below shows the number of missing data points in each column using PySpark. Pyspark needs a bit longer syntax than pandas, where we can simply write `pd.DataFrame(data.isna().sum()).T`
</p>	
```
# check if the data has missing values
data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()
```
 We can then check for the number of duplicates in the data as: 
```python
# check if the data has duplicates ... count will be more than 1 ... similar to duplicates in pandas.
data.groupBy(data.columns).count().where(col('count') > 1).show(10)
```
## Imbalance in data set
<p align='justify'>
Generally speaking, the dataset is imbalanced if the classes present in the dataset do not exist in the right proportions. In our case, we have two clear classes: defaulted nd non-defaulted cases 
</p>
``` python
# make a box plot
plt.figure(figsize = (5,3) )
sns.boxplot(x = 'loan_status', y = 'loan_percent_income', data = data_pds, color = 'magenta')

plt.xlabel('Loan Status', size = 18)
plt.ylabel('Loan % Income', size=18)
plt.show()
```
... add figure ..
<p align= 'justify'>
The box plot shows the existence of a certain amount of imbalance in the data.
</p>

## Data Exploration: Observing the sector-wise default cases
<p align = 'justify'>
Understanding the number of default cases in each sector and the typical loan amount for such cases is quite interesting. For this, we use the pandas groupby library to create sector-wise groups and find the mean and count of each aggregate. I use pandas for this wrangling step since I will use it later to plot in Seaborn, which is available only in pandas.
</p>
```python
# extract out the cases where the default has occurred for different types of loans
default_cases = data_pds[data_pds['loan_status']==1]

# Compute the mean loan amount  and the size of all the default cases for each loan type
default_summary  = default_cases.groupby('loan_intent').agg({'loan_amnt':'mean', 'loan_status':'count'})
default_summary.columns = ['avg_loan_amount', 'default_count']

# sort values of most number of default cases
default_count = default_summary.sort_values(by = 'default_count', ascending = False)
print('sector with largest principle defaulted',default_summary)

# sort the average loan amount per intent type
avgloanamount = default_summary.sort_values(by= 'avg_loan_amount', ascending = False)
print('sector with most number of default cases', avgloanamount)
```

<p align = 'justify'>
A better way to understand the count of the default cases is by making a pie chart,
as shown below. 
</p>

## Predicting Defaults using Supervised Learning.
<p align = 'justify'>
The most important step before performing any predictive analysis is to preprocess the data to prepare it for analysis.
They can have several types of imperfections like: duplicates, missing values, unnormalised data columns, to name a few. Also, they might exist in a format that is difficult to use for predictive analysis, like non-numerical classes for categorical data.  
These elements are present in nearly all raw data; hence, nearly every raw dataset needs some form of transformation to increase its usability. 
</p>

<p align = 'justify'>
We shall use the following scheme to address the shortcomings in our current dataset: <br>
1. Missing values shall be replaced with the median values since these parameters show a skewed distribution and have a significant amount of outliers. <br>
2. We remove the means of each parameter and scale the variance to 1. <br>
3. Convert  the categorical data into numerical format using One-hot Encoding.<br> 
<br>
These pre-processing steps can be very easily implemented using the <b>pipeline</b> module of scikit-learn as shown below.
</p>

```python

# Create the data sets and transform, normalise and split the data
X = data_pds[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate','person_home_ownership', 'loan_intent', 'cb_person_default_on_file']]
y = data_pds['loan_status']
X = X.drop(columns = ['loan_amnt'])

# The data that are in numbers format
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_int_rate']

# The data that is in the form of category
categorical_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']

numerical_transformer = Pipeline(steps = [
('imputer', SimpleImputer(strategy = 'median')),
('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps = [
('imputer', SimpleImputer(strategy = 'most_frequent')),    # use the mode to replace the missing values
('onehot', OneHotEncoder(handle_unknown = 'ignore'))       # one-hot encoding
])

# create a preprocessing object with all the features into it...
preprocessor = ColumnTransformer(
transformers = [
('num', numerical_transformer, numeric_features),
('cat', categorical_transformer, categorical_features)
])

X_precessed = preprocessor.fit_transform(X)
categorical_columns  = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
all_columns = numeric_features + list(categorical_columns)
```
<p align = 'justify'>
Once the data is correctly preprocessed, we can split the data into a training and a testing set, setting a particular random state for consistency.
</p>

```python
# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_precessed, y, test_size = 0.2, random_state = 42)
``` 
<p align = 'justify'>
For each ML technique, first an object is created, then fitted with the training data, and finally use the test data to benchmark its predictive power. Then, we use the metrics module to map the accuracy, confusion matrix, and the classification report. The accuracy, as the name suggests, provides the predictive accuracy. The confusion matrix provides the class-wise performance, i.e., how much each class was correctly predicted and how much the model confused the prediction with the other class. It also indicates the precision and the recall of the prediction. The accuracy of the prediction is lost either in the form of precision or in the form of recall. Precision can naievely thought of as how precise the predictions are whereas recall is how many of the actual cases were correctly identified. 
</p>
<p align = 'justify'>
Usually there is a tradeoff between the precision and the recall. However depending on the objective of the prediction we can optimise this tradeoff. For e.g., if we want to detect some critical illness, we would want the lowest possible recall. This means a large number of false positives cases might be reported. Howeever, we can preform additional screening to subsequently remove these cases. Another example can be in the field of finance where oen would want to detect fraud. Low recall is needed to catch any fraud case.
</p>

<p align = 'justify'>
I test four differernt 4 different techniques: Logistic regression, Random forests, K-Nearest neighbor classification and Support Vector machines (SVM) which perform better on imbalanced datasets. I tested 3 different kernels of SVm , the linear, polynomial (n-3) and the rbf, commonly called as the gaussian kernel. Their implementation is shown in the code below

```python 
# testing logistic regression 	
logistic_classifier = LogisticRegression(random_state =42)
logistic_classifier.fit(X_train, y_train)

ypred = logistic_classifier.predict(X_test)

print('accuracy', accuracy_score(y_test, ypred))
print('confusion matrix', confusion_matrix(y_test, ypred))
print('classification report', classification_report(y_test, ypred))

# instantiate the random forest classifier
random_forest_classifier = RandomForestClassifier(random_state = 42)
random_forest_classifier.fit(X_train, y_train)
ypred = random_forest_classifier.predict(X_test)

print('accuracy', accuracy_score(y_test, ypred))
print('confusion matrix', confusion_matrix(y_test, ypred))
print('classification report', classification_report(y_test, ypred))

# instantiate the random forest classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
ypred = knn_classifier.predict(X_test)

print('accuracy', accuracy_score(y_test, ypred))
print('confusion matrix', confusion_matrix(y_test, ypred))
print('classification report', classification_report(y_test, ypred))

# here we try the linear SVC
svm_clf = svm.LinearSVC() # linear kernal and 1:1 weights
svm_clf.fit(X_train, y_train)

ypred = svm_clf.predict(X_test)
print('accuracy', accuracy_score(y_test, ypred))
print('confusion matrix', confusion_matrix(y_test, ypred))
print('classification report', classification_report(y_test, ypred))

# here we try the poly SVC
svm_clf = svm.SVC(kernel= 'poly', degree = 3) # polynmomial kernal and 1:1 weights
svm_clf.fit(X_train, y_train)

ypred = svm_clf.predict(X_test)
print('accuracy', accuracy_score(y_test, ypred))
print('confusion matrix', confusion_matrix(y_test, ypred))
print('classification report', classification_report(y_test, ypred))

# here we try the rbf SVC
svm_clf = svm.SVC(kernel= 'rbf') # rbf and 1:1 weights
svm_clf.fit(X_train, y_train)

ypred = svm_clf.predict(X_test)
print('accuracy', accuracy_score(y_test, ypred))
print('confusion matrix', confusion_matrix(y_test, ypred))
print('classification report', classification_report(y_test, ypred))
```

<p align = 'justify'>
Finally, we can tabulate the performance of each of these methods:
</p>

	
	





 


