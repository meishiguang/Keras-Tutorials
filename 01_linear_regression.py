from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
trX = np.linspace(-1, 1, 101)
# create a y value which is approximately linear but with some random noise
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# Linear regression model
model = Sequential()
model.add(Dense(activation='linear', kernel_initializer='normal', input_dim=1, output_dim=1))
model.compile(optimizer=SGD(lr=0.01), loss='mean_squared_error', metrics=['accuracy'])

# Print initial weights
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]

# Train
model.fit(trX, trY, epochs=100, verbose=1)

#Print trained weights
weights = model.layers[0].get_weights()
w = weights[0][0][0]
b = weights[1][0]
print('Linear regression model is trained with weight w: %.2f, b: %.2f' %(w, b))

plt.plot(trX, trY, label='data')
plt.plot(trX, w_init*trX + b_init, label='init')
plt.plot(trX, w*trX + b, label='predication')
plt.legend()
plt.show()
