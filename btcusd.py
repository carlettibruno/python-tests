import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import base

loaded = base.load_data(sys.argv[1], sys.argv[2])
X = loaded[0]
y = loaded[1]

n_alphas = 100
alphas = np.logspace(-10, -2, n_alphas)

data_test = eval(sys.argv[3:][0])

#fixing the long results with "...", example: [1.222, 23.321312, ... , 2.32322]
np.set_printoptions(threshold=sys.maxsize)

coefs = []
target_pred = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

    target_pred = ridge.predict(data_test)

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