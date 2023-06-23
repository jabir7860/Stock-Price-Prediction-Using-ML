import numpy as np # Deals with Python Lists/Arrays functions of linear algebra.
import pandas as pd #Provides data structure  to work with realation & labeled data sets
import matplotlib.pyplot as plt #Graph Plot library for visualization.
import seaborn as sb #Statical Graphs in Python.

# sklearn library : 
# machine learning, pre-processing, cross-validation, and visualization 
# algorithms using a unified interface.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore') 

df = pd.read_csv('/kaggle/input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
df.head()  

df.shape #Prints how many rows & Colums are there ?   

df.describe() # ALL about data   

df.info() # About type of data & About NULL data 

#EDA - Explorartory Data Analysis 
# Analysing Data Using Visual Techniques 
plt.figure(figsize=(15,5)) 
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

df.head() 

#Check Whether Data in Close column is as same as Adj Column 
df[df['Close']==df['Adj Close']].shape #Prints how many rows are same 

# As the Data in Both COlumn are same we will drop one column 
df=df.drop(['Adj Close'],axis=1)  

#Colum is dropped we can check by 
df.head() 

#Check for NULLS are present or not in the data frame 

df.isnull().sum() 

# Distribution Plot for continuos  features given data set For each COlumn 
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
plt.show() 

#BOX PLOT 
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(df[col])
plt.show()


#From the above boxplots, we can conclude that only volume dat
#a contains outliers in it.
#but the data in the rest of the columns are free from any outlier. 


#Feature Engineering 
# Data Extraction Function TO add colums of Year/Day/Date 

splitted = df['Date'].str.split('-', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

df.head()


#We Have Added Colum Quearter End 
""""
A quarter is defined as a group of three months.
Every company prepares its quarterly results and publishes them publicly 
so, that people can analyze the company’s performance. 
These quarterly results affect the stock prices heavily which
is why we have added this feature 
because this can be a helpful feature for the learning model.
""" 
#MARCH _ JUNE - SEP - DEC 
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head() 

#YEAR WISE GRAPH PLOT 
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
plt.show()


"""
Prices are higher in the months which are quarter end as compared 
to that of the non-quarter end months.
The volume of trades is lower in the months which are quarter end
"""
df.groupby('is_quarter_end').mean()





#Adding Extra Colums to train our model best 
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0) 
df.head() 

#Pie Chart 
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()  


plt.figure(figsize=(10, 10))
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()


#Data Splitting and Normalization 
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

"""
After selecting the features to train the model on we should normalize
the data because normalized data leads to stable and fast training of the model.
After that whole data has been split into two parts with a 90/10 ratio so, 
that we can evaluate the performance of our model on unseen data.
"""

"""
Model Development and Evaluation
Now is the time to train some state-of-the-art machine learning models
(Logistic Regression, Support Vector Machine, XGBClassifier), 
and then based on their performance on the training and validation data we
will choose which ML model is serving the purpose at hand better.
"""
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print() 

""" 
Among the three models, we have trained XGBClassifier has the highest 
performance but it is pruned to overfitting as the difference between 
the training and the validation accuracy 
is too high.
But in the case of the Logistic Regression, this is not the case.
""" 

#PLOT CONFUSTION MATRIX 


"""
Conclusion:
We can observe that the accuracy achieved by the
state-of-the-art ML model is no better than simply guessing with a probability of 50%.
Possible reasons for this may be the lack of data or using a very simple model
to perform such a complex task as Stock Market prediction.

""""
