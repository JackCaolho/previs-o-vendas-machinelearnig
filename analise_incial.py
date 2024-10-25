import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Carregar os dados
df = pd.read_csv("sales.csv")

# Descrição estatística para entender as distribuições
print("\nEstatísticas descritivas:")
print(df.describe())

# Converter 'Date' para formato datetime, com o dia primeiro
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Histogramas para analisar a distribuição
df[['Weekly_Sales', 'Fuel_Price', 'Unemployment', 'Temperature', 'CPI']].hist(bins=20, figsize=(10, 8))
plt.suptitle("Distribuição das variáveis", y=1.02)
plt.tight_layout()
plt.show()

# Converter 'Date' para formato datetime
df['Date'] = pd.to_datetime(df['Date'])

# Gráficos de linha para tendências temporais
plt.figure(figsize=(10, 8))

# Tendência de vendas semanais
plt.subplot(3, 1, 1)
plt.plot(df['Date'], df['Weekly_Sales'], label='Weekly Sales', color='blue')
plt.title("Tendência de Vendas Semanais")
plt.xlabel("Data")
plt.ylabel("Vendas Semanais")
plt.grid(True)

# Tendência do Preço do Combustível
plt.subplot(3, 1, 2)
plt.plot(df['Date'], df['Fuel_Price'], label='Fuel Price', color='orange')
plt.title("Tendência do Preço do Combustível")
plt.xlabel("Data")
plt.ylabel("Fuel Price (milhares)")
plt.grid(True)

# Tendência do Desemprego
plt.subplot(3, 1, 3)
plt.plot(df['Date'], df['Unemployment'], label='Unemployment', color='green')
plt.title("Tendência do Desemprego")
plt.xlabel("Data")
plt.ylabel("Unemployment (milhares)")
plt.grid(True)

plt.tight_layout()
plt.show()


# Calcula a matriz de correlação
correlation_matrix = df.corr()

# Configurações do tamanho do gráfico
plt.figure(figsize=(10, 8))

# Gera o heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Título do gráfico
plt.title("Mapa de Calor das Correlações entre Variáveis")
plt.show()

# Pairplot para todas as variáveis numéricas
sns.pairplot(df, vars=['Weekly_Sales', 'Fuel_Price', 'Temperature', 'CPI', 'Unemployment'], diag_kind="kde")
plt.suptitle("Pairplot das Variáveis Numéricas", y=1.02)
plt.show()

# Boxplot de vendas semanais em semanas com e sem feriados
plt.figure(figsize=(8, 6))
sns.boxplot(x="Holiday_Flag", y="Weekly_Sales", data=df)
plt.xticks([0, 1], ["Sem Feriado", "Com Feriado"])  # Rótulos para melhor interpretação
plt.title("Distribuição de Vendas Semanais (Com e Sem Feriado)")
plt.xlabel("Feriado")
plt.ylabel("Vendas Semanais")
plt.show()
