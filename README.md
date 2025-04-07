# Implementation-of-Logistic-Regression-Using-Gradient-Descent

### NAME: KEERTHANA V
### REG NO: 212223220045

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required libraries and load the dataset.

2.Define X and Y array.

3.Define a function for costFunction,cost and gradient.

4.Define a function to plot the decision boundary.

5.Define a function to predict the Regression value 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KEERTHANA V
RegisterNumber: 212223220045 
```
```
import pandas as pd
import numpy as np
```
```
dataset=pd.read_csv('Placement_Data.csv')
dataset
```
## Output:
<img width="739" alt="image" src="https://github.com/user-attachments/assets/592d7c90-306c-426b-ac70-01524a07719b" />

```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
```
```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')    
dataset["status"]=dataset["status"].astype('category') 
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
## Output:
<img width="169" alt="image" src="https://github.com/user-attachments/assets/ad0b8677-0810-41bb-928c-8a0e0e890dca" />

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes   
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
## Output:
<img width="582" alt="image" src="https://github.com/user-attachments/assets/6f1b56be-3e27-495d-a405-b90f8ce907bf" />

```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
## Output:
<img width="409" alt="image" src="https://github.com/user-attachments/assets/4c156b02-167b-44cd-99cb-6f4654dfb634" />





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

