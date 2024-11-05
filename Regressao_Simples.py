import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
data = pd.read_csv('sales.csv')

# Dividir o dataset em variáveis de entrada (X) e alvo (y)
X = data[['Holiday_Flag']]
y = data['Weekly_Sales']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Exibir coeficientes do modelo
print(f"Intercepto: {model.intercept_}")
print(f"Coeficiente (Holiday_Flag): {model.coef_[0]}")
