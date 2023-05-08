import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Чтение данных из файла CSV
df = pd.read_csv('data.csv', header=None, names=['date', 'consumption'], index_col=1, parse_dates=True)

# Преобразование индекса в формат DatetimeIndex
df.index = pd.DatetimeIndex(df.index)


# Ресемплирование данных по месяцам
df_monthly = df.resample('M').sum()

x1 = np.array(df.index)
y1 = np.array(df_monthly['consumption'])
# Создание новых признаков
df_monthly['month'] = df_monthly.index.month
df_monthly['year'] = df_monthly.index.year
df_monthly['month_squared'] = df_monthly['month'] ** 2
df_monthly['year_squared'] = df_monthly['year'] ** 2

# Определение признаков и целевой переменной
X = df_monthly[['month', 'year', 'month_squared', 'year_squared']]
y = df_monthly['consumption']

plt.figure(figsize=(10,10))
plt.title("Месячное потребление электроэнергии") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()      # включение отображения сетки
plt.plot(x1,y1, 'bo-')  # Create a figure containing a single axes.
plt.show()
# Plot some data on the axes.

# Обучение модели регрессии
model = KNeighborsRegressor(n_neighbors=12)
model.fit(X, y)

# Генерация дат для предсказания на 12 месяцев вперед
start_date = df_monthly.index.max() + pd.offsets.MonthBegin(1)
end_date = start_date + pd.DateOffset(months=11)
dates = pd.date_range(start_date, end_date, freq='M')

# Создание нового DataFrame для хранения предсказаний
predictions = pd.DataFrame(index=dates, columns=['consumption'])

# Заполнение признаков в DataFrame предсказаний
predictions['month'] = predictions.index.month
predictions['year'] = predictions.index.year
predictions['month_squared'] = predictions['month'] ** 2
predictions['year_squared'] = predictions['year'] ** 2

# Предсказание потребления энергии на 12 месяцев вперед
predictions['consumption'] = model.predict(predictions[['month', 'year', 'month_squared', 'year_squared']])

x2 = np.array(dates)
y2 = np.array(predictions['consumption'])

plt.figure(figsize=(10,10))
plt.title("Предсказание месячного потребления электроэнергии") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()      # включение отображения сетки
plt.plot(x1, y1, 'bo-', x2, y2, 'ro-')  # Create a figure containing a single axes.
plt.show()
# Вывод предсказаний
print(predictions)