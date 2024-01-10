
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
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


range_leafs = range(2,25)
scores = pd.DataFrame({'Train Score': [], 'Test Score': [], 'R2 Score': [], 'Mean Absolute Error':[], 'Mean Squared Error':[], 'Mean Absolute Percentage Error': []})
scores.index.name = 'Max Leaf Nodes'
for i in range_leafs:
    model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.loc[i] = [round(model.score(X_train, y_train),4), round(model.score(X_test, y_test),4), round(r2_score(y_test, y_pred),4), round(mean_absolute_error(y_test, y_pred),4), round(mean_squared_error(y_test, y_pred),4), round(mean_absolute_percentage_error(y_test, y_pred),4)]
    
print(scores)


plt.figure(figsize=(10,5))
plt.plot(range_leafs, scores['Train Score'], color='red', label='Train Score')
plt.plot(range_leafs, scores['Test Score'], color='blue', label='Test Score')
plt.legend()
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Score')
plt.title('Max Leaf Nodes vs Score')
plt.show()


range_depth = range(1,25)
scores = pd.DataFrame({'Train Score': [], 'Test Score': [], 'R2 Score': [], 'Mean Absolute Error':[], 'Mean Squared Error':[], 'Mean Absolute Percentage Error': []})
scores.index.name = 'Max Depth'
for i in range_depth:
    model = DecisionTreeRegressor(max_depth=i, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.loc[i] = [round(model.score(X_train, y_train),4), round(model.score(X_test, y_test),4), round(r2_score(y_test, y_pred),4), round(mean_absolute_error(y_test, y_pred),4), round(mean_squared_error(y_test, y_pred),4), round(mean_absolute_percentage_error(y_test, y_pred),4)]
    
print(scores)


plt.figure(figsize=(10,5))
plt.plot(range_depth, scores['Train Score'], color='red', label='Train Score')
plt.plot(range_depth, scores['Test Score'], color='blue', label='Test Score')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Max Depth vs Score')
plt.show()


decision_tree = DecisionTreeRegressor(random_state=42, max_depth=15, max_leaf_nodes=22)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df.to_csv('DT.csv')


print('-----------------------------------')
print('Decision Tree Regressor')
print(f'Max Depth: {decision_tree.max_depth}')
print(f'Max Leaf Nodes: {decision_tree.max_leaf_nodes}')
print('Train Score: ', decision_tree.score(X_train, y_train))
print('Test Score: ', decision_tree.score(X_test, y_test))
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


