import numpy as np
from matplotlib import pyplot as plt
import LinRegression
from sklearn import datasets


# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

lr = LinRegression()
lr.fit(X, y, learning_rate=0.997, random_state=0, weight_low=-900, weight_high=900, nsteps=3000)
xx = [i for i in range(X.shape[0])]
y1 = lr.predict(X)
print 'MSE1 (My LR model):', mean_squared_error(y, y1)
f = 0
t = 40
plt.plot(xx[f:t], y[f:t], color='r', linewidth=4, label='y')
plt.plot(xx[f:t], y1[f:t], color='b', linewidth=2, label='predicted y')
plt.ylabel('Target label')
plt.xlabel('Line number in dataset')
plt.legend(loc=4)
plt.show()

