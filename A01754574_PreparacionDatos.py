import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tt
from sklearn.metrics import r2_score as r
from sklearn.metrics import mean_squared_error as ms
from sklearn.metrics import mean_absolute_error as ma


data = pd.read_csv("/home/yayo/Documentos/Programación/Python/CienciaDeDatos/DatosComida.csv", encoding="utf-8", encoding_errors="ignore")

print(data.shape)
print(data.head())
print(data.describe())


x = ['Calorias (kcal)', 'Carbohidratos (g)', 'Lipidos (g)', 'Proteina (g)', 'Sodio (mg)']
X =  data[x]

Y = data['Calorias (kcal)']

x, X, y, Y = tt(X, Y, test_size = 20, random_state = 45936)

model = lr()
model.fit(x, y)

predict = model.predict(X)

print(f'Entrenamiento: {model.score(x, y)}')
print(f'Test: {model.score(X, Y)}')

print(model.intercept_, model.coef_) 

print(f'MAE: {ma(Y, predict)}')
print(f'MSE: {ms(Y, predict)}')

def plot(X, Y, model):
    y_pred = model.predict(X)
    data = pd.DataFrame({'cal actual':Y,
                        'cal predecidas':y_pred})
    plt.figure(figsize=(12,8))
    plt.scatter(data.index,data['cal actual'].values,label='Calorias actuales')
    plt.scatter(data.index,data['cal predecidas'].values,label='Calorias Predecidas')
    plt.title('Grafica',
             fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best')
    
    plt.show()

plot(X,Y,model)




