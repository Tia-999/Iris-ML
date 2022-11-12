

# libraries to manipulate the data we are importing
import numpy as np
import matplotlib.pyplot as plt

#library where you are importing the dataset from
from sklearn.datasets import load_iris

#importing the data
dataset = load_iris()

#data in form of description
dataset


#various attributes of data
print(dataset.DESCR)



#input
X = dataset.data

#target contains their category
y = dataset.target
dataset


y

X

#Visualizing and plotting the data
plt.plot(X[:,0][y == 0]*X[:,1][y == 0],X[:,1][y == 0]*X[:,2][y == 0],'r.',label='Setosa')
plt.plot(X[:,0][y == 1]*X[:,1][y == 1],X[:,1][y == 1]*X[:,2][y == 1],'r.',label='Versicolor',color='blue')
plt.plot(X[:,0][y == 2]*X[:,1][y == 2],X[:,1][y == 2]*X[:,2][y == 2],'r.',label='Virginica',color='green')
plt.legend()
plt.show()


#Standardize/Scaling the data
from sklearn.preprocessing import StandardScaler 
X = StandardScaler().fit_transform(X)


#importing train_test_split
from sklearn.model_selection import train_test_split



#training and testing the data
X_train ,X_test, y_train, y_test =train_test_split(X,y)




from sklearn.linear_model import LogisticRegression



#Logistic Regression as our ML Model
log_reg = LogisticRegression()



#training the model
log_reg.fit(X_train,y_train)

#predict the score on test set
log_reg.score(X_test,y_test)

#predict the score on whole data set
log_reg.score(X,y)


