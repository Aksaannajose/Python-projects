import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score
boston = datasets.load_boston(return_X_y=False)
X = boston.data
y = boston.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
predicted = reg.predict(X_test)
# Regression coefficient
print('Coefficients are:\n', reg.coef_)
# Intecept
print('\nIntercept : ', reg.intercept_)
# variance score: 1 means perfect prediction
print('Variance score: ', reg.score(X_test, y_test))
# Mean Squared Error
print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
# Original data of X_test
expected = y_test
# Plot a graph for expected and predicted values
plt.title('BOSTON Dataset')
plt.scatter(expected, predicted, c='b', marker='.', s=36)
plt.plot([0, 50], [0, 50], '--r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()