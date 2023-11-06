


#This is the computation of simple  linear regression machine learning algorithm. The algorithm consists
#of reading the dataset, plotting the necessary graphs, plotting the regression line,
#finding the coefficients amd finally predicting the price of a house.

#The Libraries imported for the algorithm.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn packages for the algorithm.
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

#Code used to read a file.
df = pd.read_csv('houseprice_data.csv')

#Code used to check the correlations of the independent variable with the dependent variable.
print(df.corr())

#Code used to allocate the values of the chosen variables for ploting.
X = df.iloc[:, [3]].values
y = df.iloc[:, 0].values


#Code for the linear regression function. 
myregression = LinearRegression()
myregression.fit(X, y)


#Code used to visualise the main dataset of the scattered plot.
main1, lab = plt.subplots()

lab.scatter(X, y, color='green')

lab.plot(X, myregression.predict(X), color = 'blue')

lab.set_xlabel('Sqft_living')
lab.set_ylabel('price')


main1.tight_layout()
main1.savefig('Initialplot.png')

#Code used to fit the linear regression.
myregression = LinearRegression()
myregression.fit(X, y)


# code used to split the datasets into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, 
		random_state=0)


# code used fit the linear regression line to the training dataset:
myregression = LinearRegression()
myregression.fit(X_train, y_train)


# Code used to compute the Coefficients for the regression model.
print('Coefficients: ', myregression.coef_)
# Code used to compute the Intercept for the regression model.
print('Intercept: ', myregression.intercept_)
# code used to compute the Mean Squared for the regression model.
print('Mean squared error: %.4f'
	% mean_squared_error(y_test, myregression.predict(X_test)))
# Code used to compute the R Squared for the regression model.
print('Coefficient of determination: %.2f'
	% r2_score(y_test, myregression.predict(X_test)))

#Code used in predciting the price of a house with 1400 square ft. From the dataset
print('The Price of a House: ', myregression.predict(np.array([[1400]])))


# Code used to visualize the plots of the training datasets
main2, lab2 = plt.subplots()

lab2.scatter(X_train, y_train, color='red')
lab2.plot(X_train, myregression.predict(X_train), color='brown')

lab2.set_xlabel('Sqft_living')
lab2.set_ylabel('price')


main2.tight_layout()
main2.savefig('trainsetplot.png')


# Code used to visualize the plot of the test datasets
main3,lab3 = plt.subplots()

lab3.scatter(X_test, y_test, color='blue')
lab3.plot(X_test, myregression.predict(X_test), color='red')

lab3.set_xlabel('Sqft_living')
lab3.set_ylabel('price')


main3.tight_layout()
main3.savefig('testsetplot.png')



