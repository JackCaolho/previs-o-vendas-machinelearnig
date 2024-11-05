import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
data = pd.read_csv('sales.csv')

# Ate njuesa coluna 'Date' para o tipo datetime com o formato correto
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Extrair características da data
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek  # 0=segunda, 6=domingo

# Ajuste nas colunas Fuel_Price e Unemployment
data['Fuel_Price'] = data['Fuel_Price'] / 1000
data['Unemployment'] = data['Unemployment'] / 1000

# Selecionar variáveis explicativas e variável alvo
X = data[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'DayOfWeek']]
y = data['Weekly_Sales']

# Dividir os dados em conjuntos de treino e teste
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

# Exibir os coeficientes do modelo
print(f"Intercepto: {model.intercept_}")
for i, col in enumerate(X.columns):
    print(f"Coeficiente para {col}: {model.coef_[i]}")
