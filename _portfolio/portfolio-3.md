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
What is quite interesting is to understand the number of  default cases in each sector and the typical loan amount for such cases. For this, we use the pandas groupby library to create sector-wise groups and find the mean and count of each aggregate. I use pandas for this wrangling step since I will use it later to plot in seaborn, which is available only in pandas.
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
Now we visualise this data as
</p>

<p align = 'justify'>
I think a better way to understand the count of the default cases is by making a pie chart,
as shown below.... 
</p>






 


