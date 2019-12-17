import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Zaimportowanie datasetu
dataset = pd.read_csv('salary_data.csv')
X = dataset.iloc[:, :-1].values #pobranie kopii zestawu danych z wyłączeniem ostatniej kolumny
y = dataset.iloc[:, 1].values #pobranie tablicy zestawu danych z kolumny 1

# Podzielenie zestawu danych na zestaw szkoleniowy i testowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Dopasowanie prostej regresji liniowej do zestawu treningowego
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Wizualizacja wyników zestawu treningowego
viz_train = plt
viz_train.scatter(X_train, y_train, color='pink')
viz_train.plot(X_train, regressor.predict(X_train), color='yellow')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

# Wizualizacja wyników zestawu testowego
viz_test = plt
viz_test.scatter(X_test, y_test, color='green')
viz_test.plot(X_train, regressor.predict(X_train), color='grey')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()