# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VAISHNAVI S
RegisterNumber: 212222230165  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

# splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# displaying predicted values
y_pred

plt.scatter(x_train,y_train,color='brown')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='purple')
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE= ",mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
# df.head()
![Screenshot 2023-10-05 093851](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/3e171905-1a2b-422f-b1aa-95971e62c9a4)

#  df.tail()
![Screenshot 2023-10-05 093857](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/d0cdf535-17df-4ddd-ae02-cd9090378e76)

# Array value of X

![Screenshot 2023-10-05 093904](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/20673ad5-b4b0-46bb-88e1-225fe82aceb9)

# Array value of Y
![Screenshot 2023-10-05 093915](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/9d8300fd-cf28-4bcf-a4ad-a763068bb660)

# Values of Y prediction
![Screenshot 2023-10-05 094729](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/5ec7fcda-50d9-431f-adca-513cc612af02)


#  Array values of Y test
![Screenshot 2023-10-05 094736](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/f1319824-d658-4633-b01d-a34a74b3739f)


# Training Set Graph
![Screenshot 2023-10-05 094741](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/a57595d5-5042-4b63-bf4b-f7b7068f0cf9)

# Test Set Graph

![Screenshot 2023-10-05 094746](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/0a01b8c8-2116-4f28-bcbb-f2d09df4a63e)

# Values of MSE, MAE and RMSE
![Screenshot 2023-10-05 094751](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/907e0588-9aae-4a5e-befd-cf694116e924)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
