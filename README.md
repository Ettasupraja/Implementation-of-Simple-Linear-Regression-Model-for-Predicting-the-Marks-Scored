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
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())

![image](https://github.com/user-attachments/assets/9070928c-1030-4667-adb7-3642a7d7ceac)

df.info()
![image](https://github.com/user-attachments/assets/bfac0c94-c352-408f-9f9f-bf636ffbc701)

x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

![image](https://github.com/user-attachments/assets/a4bd4cfa-3a23-425f-913c-296d6bbd7619)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape

![image](https://github.com/user-attachments/assets/3bb0247a-5b2d-467c-93e7-22a799d6a978)

x_test.shape
![image](https://github.com/user-attachments/assets/3aed00f0-c925-47b7-9dab-9e2db146222f)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

![image](https://github.com/user-attachments/assets/7c553872-c08b-44fe-85c8-9ea590bda75e)

y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
![image](https://github.com/user-attachments/assets/3a3c50b3-4dfc-427a-907b-0448b890802d)

plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)') 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()         


```

## Output:
![image](https://github.com/user-attachments/assets/c7903140-c739-48b1-b64b-afba34dd8a7c)

![image](https://github.com/user-attachments/assets/7a2a7d02-fa01-4544-89c8-6eef264b064c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
