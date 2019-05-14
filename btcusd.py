# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause

#print(__doc__)

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import base

loaded = base.load_data('', 'dataset_crypto.csv')
X = loaded[0]
y = loaded[1]

# X is the 10x10 Hilbert matrix
#X = 1. / (np.arange(1, 5) + np.arange(0, 4)[:, np.newaxis])
#X = [[1.0,0.5,0.33333333,6],[0.5,0.33333333,0.25,5],[0.33333333,0.25,0.2,4],[1,2,3,4]]

#print(X)
#y = np.ones(10)
#y = [10, 0, 40, 10]
#print(y)

#print('#######################################')

# #############################################################################
# Compute paths

n_alphas = 100
alphas = np.logspace(-10, -2, n_alphas)
#print(alphas)

# print(sys.argv[1:][0])
data_test = eval(sys.argv[1:][0])
# print(data_test)

#fixing the long results with "...", example: [1.222, 23.321312, ... , 2.32322]
np.set_printoptions(threshold=sys.maxsize)

coefs = []
target_pred = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

    #data_test = [[0.0,-10.1623,-50.0870,-10.7567,-0.4245],[0.0,-10.1623,-50.0870,-0.7567,-0.2]] #21:56
    #data_test = [[sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]]]
    #print(sys.argv[1:])

    #data_test = [[float(i) for i in sys.argv[1:]]]
    # print(sys.argv[1:][0])
    # data_test = eval(sys.argv[1:][0])
    # print(data_test)

    #print(data_test)
    target_pred = ridge.predict(data_test)
    #print(' ============================ predict: ')
    #print(target_pred)
    #print(ridge.coef_)

# #############################################################################
# Display results

print(target_pred)

# ax = plt.gca()

# ax.plot(alphas, coefs)
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show() 