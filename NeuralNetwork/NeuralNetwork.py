
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf


data = pd.read_csv('./Data/boston.csv')

data.head()


X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


input_shape = X_train.shape[1]

input_shape


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')



epochs = 100
batch_size = 8

results = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))


error = model.evaluate(X_test, y_test, verbose=1)

print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))


y_pred = model.predict(X_test)

# y_pred, y_test


# metrics 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('TensorFlow Model')
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2: ', r2_score(y_test, y_pred))


plt.title('Loss Curves')
plt.plot(results.history['loss'], label='Train')
plt.plot(results.history['val_loss'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
print('Epochs: ', len(results.epoch))
print(results.history.keys())


