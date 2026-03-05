import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = pd.read_csv("../lin-reg/Linear_X_Train.csv").values
y = pd.read_csv("../lin-reg/Linear_Y_Train.csv").values

theta = np.load("../lin-reg/Theta_list.npy")

plt.ion()

T0 = theta[:, 0]
T1 = theta[:, 1]

for i in range(0, 50):
    y_ = T1[i]*X + T0
    plt.scatter(X, y)
    plt.plot(X, y_, color = "red")
    plt.draw()
    plt.pause(0.1)
    plt.clf()