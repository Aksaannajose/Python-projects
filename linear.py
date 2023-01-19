import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.array([1,5,1,9,33,2]).reshape((-1, 1))
y = np.array([2,7,1,9,3,40])
print(x)
print(y)
model=LinearRegression()
model.fit(x, y)
r_sq=model.score(x,y)
print('Coefficient od determination: ', r_sq)
print('Intercept: ', model.intercept_)
print('Slope: ', model.coef_)
y_pred=model.predict(x)
plt.plot(x, y_pred, color="r")
plt.xlabel('x')
plt.ylabel('y')
plt.show()