import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
data = pd.read_csv('sales.csv')

# Converter a coluna 'Date' para o tipo datetime com o formato correto
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

# Inicializar o modelo de Gradient Boosting
model = GradientBoostingRegressor(random_state=42)

# Definir os hiperparâmetros a serem ajustados
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

# Realizar busca em grade para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Exibir os melhores hiperparâmetros
print(f"Melhores Parâmetros: {grid_search.best_params_}")

# Fazer previsões no conjunto de teste
y_pred = grid_search.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Calcular RMSE
rmse = mse ** 0.5
print(f"Root Mean Squared Error: {rmse}")
