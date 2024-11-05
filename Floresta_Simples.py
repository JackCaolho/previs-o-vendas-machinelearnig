import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('sales.csv')

# Ajuste na coluna 'Date' para o tipo datetime com o formato correto
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
X = data[['CPI', 'Month', 'Day']]
y = data['Weekly_Sales']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir o modelo de Random Forest
model = RandomForestRegressor(random_state=42)

# Definir os parâmetros para busca em grade
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Realizar a busca em grade
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Realizar validação cruzada
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
mse_cv = -cv_scores.mean()
r2_cv = best_model.score(X_test, y_test)  # R² do conjunto de teste

print(f"Melhores Parâmetros: {grid_search.best_params_}")
print(f"Mean Squared Error (Validação Cruzada): {mse_cv}")
print(f"R^2 Score (Teste): {r2_cv}")

# Fazer previsões no conjunto de teste
y_pred = best_model.predict(X_test)

# Avaliar o modelo
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Teste): {mse_test}")
print(f"R^2 Score (Teste): {r2_test}")

# Avaliar a importância das variáveis
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# Visualizar a importância das variáveis
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importância')
plt.title('Importância das Variáveis no Modelo Random Forest')
plt.show()
