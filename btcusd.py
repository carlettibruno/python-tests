# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import base

loaded = base.load_data('.', 'Bitfinex_BTCUSD_worked.csv')
print(loaded)

# X is the 10x10 Hilbert matrix
#X = 1. / (np.arange(1, 5) + np.arange(0, 4)[:, np.newaxis])
X = [[1.0,0.5,0.33333333,6],[0.5,0.33333333,0.25,5],[0.33333333,0.25,0.2,4],[1,2,3,4]]

print(X)
#y = np.ones(10)
y = [10, 0, 40, 10]
print(y)

print('#######################################')

# #############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
print(alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

    data_test = [[1.0,0.5,0.33333333,6],[0.5,0.33333333,0.25,5],[0.33333333,0.25,0.2,4],[0.25,0.2,0.16666667,2]]
    target_pred = ridge.predict(data_test)
    print(' ============================ predict: ')
    print(target_pred)
    #print(ridge.coef_)

# #############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()