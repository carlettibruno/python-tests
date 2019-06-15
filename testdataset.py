import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(basis_X_train, basis_y_train,
                      basis_X_test,basis_y_test):
    
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(basis_X_train, basis_y_train)
    # Make predictions using the testing set
    basis_y_pred = regr.predict(basis_X_test)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(basis_y_test, basis_y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(basis_y_test,
                                            basis_y_pred))
    # Plot outputs
    plt.scatter(basis_y_pred, basis_y_test,  color='black')
    plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)
    plt.xlabel('Y(actual)')
    plt.ylabel('Y(Predicted)')
    plt.show()
    
    return regr, basis_y_pred
    

def create_features(data):
    basis_X = pd.DataFrame(data)
    basis_X.fillna(0)

    basis_y = data['target']
    basis_y.dropna(inplace=True)                        

    print("Any null data in y: %s, X: %s"
            %(basis_y.isnull().values.any(), 
             basis_X.isnull().values.any()))
    print("Length y: %s, X: %s"
            %(len(basis_y.index), len(basis_X.index)))
    
    return basis_X, basis_y    


#data = pd.read_csv("") 

#24/04/2019 02:53
dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
data = pd.read_csv('/Users/brunocarletti/Downloads/dataset_crypto.csv', parse_dates=[0], date_parser=dateparse)
data.set_index('datetime', inplace=True)
print(data.head())
data.plot(figsize=(15,8))
plt.show()

training_data = data[:3500]
print('training_data')
print(training_data)
validation_data = data[3500:]
print('validation_data')
print(validation_data)

basis_X_train, basis_y_train = create_features(training_data)
basis_X_test, basis_y_test = create_features(validation_data)    

print('basis_X_train')
print(basis_X_train)
print('basis_y_train')
print(basis_y_train)
print('basis_X_test')
print(basis_X_test)
print('basis_y_test')
print(basis_y_test)
_, basis_y_pred = linear_regression(basis_X_train,basis_y_train,basis_X_test,basis_y_test)

# ax = plt.gca()

# ax.plot(alphas, coefs)
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show() 
