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
Developed by: ETTA SUPRAJA
RegisterNumber: 212223220022 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/


```

## Output:

DATA SET
![image](https://github.com/user-attachments/assets/b945394a-2f01-4aa1-93b2-f001c896ec7b)

HEAD VALUES
![image](https://github.com/user-attachments/assets/5e1f1940-81b3-4a6d-a044-1bf417e7b019)

TAIL VALUES
![image](https://github.com/user-attachments/assets/3f410fe0-e1bc-495f-8b05-a00cf01dbff6)

X and Y VALUES
![image](https://github.com/user-attachments/assets/a9f523f4-d32c-4edb-a56a-51c6c8cad430)

Prediction Values of X and Y
![image](https://github.com/user-attachments/assets/f3781f62-4552-474a-9fee-2189f7c3d77f)

GRAPH
![image](https://github.com/user-attachments/assets/e76e3e66-c594-416f-990d-331b4322c9c5)
![image](https://github.com/user-attachments/assets/dc08481b-3b1b-4f19-82ba-9b61948293fc)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
