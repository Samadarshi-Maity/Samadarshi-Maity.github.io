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
Here, I analyze a credit risk dataset to check if I can use supervised learning strategies for predicting the chances of default. I will first load and explore the data using Pyspark. Pyspark also also easy data exploration like pandas. Then we will use sk-learn to perform some supervised learning. I will particularly try Logistic Regression Classifier, bagged tree classsfier using random Forests, K-Nearest Neighbor Search and Support Vector Machines for predicting the defaulting cases. I have added inline codes to describe the process but you can find a notebook that 'RUN-ready' on google colab <a href '...'>here</a>.
</p>
<p align="justify">
First, we import all the necessary packages 
for data manipulation: pypspark, pandas and numpy, Plotting tools: matplotlib and seaborn, supervised ML: scikit-learn-LogisticRegression,  KNeighborsClassifiers, SVM (linear, poly and rbf). We test the quiality of the predictions using these techniques: sklearn-metrics. 
Before, aplying ML operations, the data has to be preprocessed to make it ready for analysis. For this, we use imputing, column transformation and piplining modules of scikit-learn. 
</p>
  ``` python
  # import some packages here
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#  import spark session to create a session
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
