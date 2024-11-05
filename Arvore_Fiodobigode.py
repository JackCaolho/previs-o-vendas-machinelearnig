import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Carregar os dados
data = pd.read_csv('sales.csv')

# Ajuste nas colunas Fuel_Price e Unemployment
data['Fuel_Price'] = data['Fuel_Price'] / 1000
data['Unemployment'] = data['Unemployment'] / 1000

# Selecionar variáveis explicativas e variável alvo
# Removendo colunas de baixa importância, se necessário
X = data[['Holiday_Flag', 'Temperature', 'CPI']]  # Por exemplo, removendo 'Fuel_Price' e 'Unemployment'
y = data['Weekly_Sales']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir o modelo de Árvore de Decisão
model = DecisionTreeRegressor(random_state=42)

# Definir os parâmetros para busca em grade
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
}

# Realizar a busca em grade
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
y_pred = best_model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Melhores Parâmetros: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Avaliar a importância das variáveis
feature_importances = best_model.feature_importances_
features = X.columns  # ou use as colunas originais se não normalizou
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualizar a importância das variáveis
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importância')
plt.title('Importância das Variáveis')
plt.show()

# Visualizar a árvore de decisão
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=features)
plt.title('Árvore de Decisão')
plt.show()
