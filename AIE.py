from Models import *
from Training import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET = 'C:/Users/Prometeo/Desktop/Marco/PrometheanRobotics/AIE Project/datasets'

# function to Normalize the inputs in a range (0 - 1)
def Normalize(X):
    for i in range(len(X[0])):
        x = X[:, i]
        M = max(x)
        m = min(x)
        for row in X:
            if np.isnan(row[i]):
                row[i] = 0
            row[i] = (row[i] - m) / (M - m)
    return X

# importing dataset
X = pd.read_csv(DATASET+'/housing/housing.csv')
X = X[:][['housing_median_age', 'total_rooms', 'total_bedrooms', 'median_income', 'median_house_value']]
X = np.array(X)
X = Normalize(X)
y = X[:, 4]
X = X[:, :4]

# model graph and instance creation
LG = [[4, 'relu'], [15, 'relu'], [1, 'tanh']]
modello = Model(LG, 1)
print(modello.G)

# training algorithm
history = fit(modello, X, y, valsplit=0.005, epochs=400, batchsize=1, loss='mse', momentum=0.5, learning_rate=0.05,
              earlystopping=True, patience=3, reduceonplateu=True, reducefactor=0.8, metrics=['accuracy'])

# metrics history plot
print(history)
pd.DataFrame(history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.5)
plt.show()