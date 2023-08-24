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
![Screenshot 2023-08-24 090506](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/10870dc6-ba5b-468e-af3f-1e31b22c1f35)

#  df.tail()
![Screenshot 2023-08-24 090513](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/0a2c55fc-f843-4138-b37d-d3c99a4b0d35)

# Array value of X
![Screenshot 2023-08-24 090520](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/ba2cb443-fcce-435a-b2a3-c81bec14c885)

# Array value of Y
![Screenshot 2023-08-24 090529](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/fc426798-a55b-4dc4-94e9-bb8ba03232f2)

# Values of Y prediction
![Screenshot 2023-08-24 090540](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/625c9e19-5a58-452a-8b85-a6b7616e55de)

#  Array values of Y test


# Training Set Graph
![Screenshot 2023-08-24 090549](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/0cb0ce0d-362d-4764-8627-ce44f2d2298b)

# Test Set Graph
![Screenshot 2023-08-24 090558](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/0c2c108c-4724-4240-80c8-f3cd2b88b640)

# Values of MSE, MAE and RMSE
![Screenshot 2023-08-24 090605](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/64f372d2-8885-4a2a-a277-a1bcb35594c0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
