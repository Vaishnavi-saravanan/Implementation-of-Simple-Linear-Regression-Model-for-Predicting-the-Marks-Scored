# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
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
![Screenshot 2023-08-24 090506](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/525e207b-4ef9-42d9-89d3-b06d8fc5946b)
![Screenshot 2023-08-24 090513](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/541e04e1-279c-4fa9-8b42-0f3df88985b7)
![Screenshot 2023-08-24 090520](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/a617b3b2-deb5-4427-a39d-763c8e37bf4e)
![Screenshot 2023-08-24 090529](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/a7851792-4ff0-4187-a992-fc18747d179e)
![Screenshot 2023-08-24 095130](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/f0e8ff61-5231-4fb2-a6d9-c4b7b24ba163)
![Screenshot 2023-08-24 095137](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/c340470b-1ed8-4dbb-af36-e215ee6d8853)


![Screenshot 2023-08-24 095144](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/82c79516-9123-4978-b236-48cc5aac6864)



![Screenshot 2023-08-24 090549](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/55131d4e-4721-499b-85c6-0b4fb000659a)
![Screenshot 2023-08-24 090558](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/c34fef43-115d-4cef-b60c-de546449c2ce)
![Screenshot 2023-08-24 090605](https://github.com/Vaishnavi-saravanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118541897/1ae04315-b235-4bcf-917b-21f37e81b6d9)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
