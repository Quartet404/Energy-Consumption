import csv
import copy
import datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense
filename = "Energy Consumption.csv"

x=np.array([])
y=np.array([])
with open(filename, "r", newline="") as file:
    reader = csv.reader(file)
    reader.__next__()
    for row in reader:
        x = np.append(x, dt.datetime.strptime(row[1], "%Y-%m-%d"))
        y = np.append(y, float(row[2]))

print("done")

plt.figure(figsize=(10,10))
plt.title("Месячное потребление электроэнергии") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()      # включение отображения сетки
plt.plot(x,y, 'bo-')  # Create a figure containing a single axes.
# Plot some data on the axes.

plt.show()

model = Sequential()
model.add(Dense(units = 64, activation = 'relu', input_dim=24))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 4, activation = 'relu'))
model.add(Dense(units = 1, activation = 'relu'))
model.compile(optimizer='adam', loss='mse')









input()
exit()
