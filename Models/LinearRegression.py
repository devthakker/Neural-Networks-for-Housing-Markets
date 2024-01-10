
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


# data 
# https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data


data = pd.read_csv('./Data/boston.csv')

data.info()

data.dropna(inplace=True)



corr_matrix = data.corr()

corr_matrix


sns.heatmap(corr_matrix, annot=True)
plt.show()


X = data.drop(['MEDV'], axis=1)

y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df.to_csv('LR.csv')

df


print('-----------------------------------')
print('Linear Regression')
print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))
print('Intercept: ', regressor.intercept_)
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('R2 Score: ', r2_score(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test, y_pred))
# print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
print('-----------------------------------')


plt.scatter(y_test, y_pred, color='grey', marker='o')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
a, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, a*y_test+b)        
plt.show()









