import csv
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
filename = "Energy Consumption.csv"

def deriv(x):
    x = copy.deepcopy(x)
    if(type(x)==np.ndarray):
        if(type(x[0])==np.float64):
            x1 = np.roll(x,1)
            return np.insert(((x1-x)*(x.size/100))[1:], 0, 0.0)
    else:
        print("Wrong type")

x=np.array()
y=np.array()
with open(filename, "r", newline="") as file:
    reader = csv.reader(file)
    reader.__next__()
    for row in reader:
        x.append(row[1])
        y.append(float(row[2]))
print("done")

plt.title("Месячное потребление электроэнергии") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()      # включение отображения сетки
plt.plot(x,y, 'bo-')  # Create a figure containing a single axes.
# Plot some data on the axes.

plt.show()
input()
exit()
