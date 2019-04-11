import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression()
data_train = [[4000], [5000], [6000], [5500], [7000], [1000], [8000], [11000], [12000], [5100], [2600], [3500]]
#data_train = data_set_train[:, np.newaxis, 2]
print(data_train)
target_train = [100, 120, 130, 110, 130, 50, 140, 155, 160, 80, 55, 60]

data_test = [[7000], [5000], [6000]]
target_test = [119, 97, 108]
reg.fit(data_train, target_train)

print(reg.coef_)

target_pred = reg.predict(data_test)
print(target_pred)

plt.scatter(data_test, target_test,  color='black')
plt.plot(data_test, target_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()