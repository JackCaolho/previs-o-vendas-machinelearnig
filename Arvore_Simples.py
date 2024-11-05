import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
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
X = data[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'DayOfWeek']]
y = data['Weekly_Sales']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicializar e treinar o modelo de Árvore de Decisão
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Exibir a importância das variáveis
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nImportância das Variáveis:")
print(importance_df)

# Plotar a importância das variáveis
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Importância")
plt.title("Importância das Variáveis na Árvore de Decisão")
plt.show()

# Exibir a profundidade da árvore para referência
print(f"Profundidade da Árvore: {model.get_depth()}")

# Visualizar a árvore de decisão
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=features, filled=True, rounded=True)
plt.title("Visualização da Árvore de Decisão")
plt.show()
