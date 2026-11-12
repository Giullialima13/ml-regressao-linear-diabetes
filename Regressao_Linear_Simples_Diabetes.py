# REGRESSÃO LINEAR SIMPLES DATASET DIABETES (scikit-learn)
# Apresentação do algoritmo:
# A Regressão Linear Simples é um algoritmo de aprendizagem supervisionada que prevê 
# um valor numérico contínuo (y) modelando a relação com uma variável independente (x) 
# através de uma reta: y = b0 + b1 * x. Os coeficientes são estimados por mínimos quadrados.

# Apresentação do dataset e do objetivo:
# Dataset Diabetes (scikit-learn): 442 doentes.
# Objetivo: prever a PROGRESSÃO DA DOENÇA (y) a partir do IMC (x).

# 1. Importar as bibliotecas
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 2. Carregar os dados (valores reais, sem normalização)
diabetes = load_diabetes(scaled=False)
X = diabetes.data[:, [2]]  # Coluna 2: Índice de Massa Corporal (IMC) em formato 2D
y = diabetes.target          # Variável alvo: progressão da doença

# 3. Dividir em treino (80%) e teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# 5. Fazer previsões com os dados de teste
y_previsto = modelo.predict(X_teste)

# 6. Avaliar o modelo
r2 = r2_score(y_teste, y_previsto)
mae = mean_absolute_error(y_teste, y_previsto)

# 7. Gráfico: pontos reais e reta de regressão
plt.figure(figsize=(6, 4))
plt.scatter(X, y, alpha=0.5, label="Dados reais")
plt.plot(X, modelo.predict(X), color="red", label="Reta de regressão")
plt.xlabel("IMC (kg/m^2)")
plt.ylabel("Progressão da Doença")
plt.title("Regressão Linear Simples - Dataset Diabetes")
plt.legend()
plt.show()

# 8. Resultados numéricos
b0 = modelo.intercept_
b1 = modelo.coef_[0]

previsao_imc30 = modelo.predict([[30]])[0]

print(f"Equação da reta: y = {b0:.4f} + {b1:.4f} * x")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Previsão para um doente com IMC=30: {previsao_imc30:.2f}")

# 9. Análise dos pontos (resíduos = valor real - valor previsto)
residuos = y_teste - y_previsto
bem_previstos = sum(abs(residuos) <= 25)
mal_previstos = sum(abs(residuos) > 100)

# Identificar o doente mais afastado da reta
import numpy as np
indice_mais_afastado = np.argmax(abs(residuos))
erro_maximo = residuos[indice_mais_afastado]

print(f"Doentes com erro <= 25: {bem_previstos}")
print(f"Doentes com erro > 100: {mal_previstos}")
print(f"Doente mais afastado da reta (resíduo): {erro_maximo:.2f}")

# 10. Interpretação do gráfico e resultados
print("\n--- INTERPRETAÇÃO DOS RESULTADOS ---")
print("1. A reta vermelha representa a tendência linear estimada que relaciona o IMC com a progressão da doença. "
      f"O b0 ({b0:.2f}) é o valor previsto de y se o IMC fosse 0 (teórico), e o b1 ({b1:.2f}) indica que, "
      "em média, por cada unidade a mais de IMC, a progressão da doença aumenta cerca de 10.8 pontos.")
print("2. Cada ponto azul representa um doente real do dataset. Estar acima da reta significa que a doença "
      "evoluiu mais do que o modelo previa (subestimação); estar abaixo significa que evoluiu menos (sobrestimação).")
print("3. Os pontos devem estar próximos da reta para garantir previsões rigorosas. Os melhores pontos têm resíduos próximos de zero.")
print(f"No conjunto de teste, temos {bem_previstos} doentes bem previstos (erro <= 25) e {mal_previstos} mal previstos (erro > 100).")
print("4. Os pontos distribuem-se numa nuvem ascendente. A relação entre o IMC e a progressão é positiva, mas fraca a moderada.")
print(f"5. Com um R2 de aproximadamente {r2:.2f}, o modelo explica apenas cerca de 23% da variação, pelo que não deve ser "
      "usado isoladamente para diagnósticos individuais precisos. Para melhorar, deve-se usar regressão múltipla (incluindo as outras 9 variáveis). "
      f"O MAE é de {mae:.2f}, o que significa que as previsões apresentam, em média, um erro absoluto de aproximadamente {mae:.2f} pontos.")
print("6. Foi utilizada a Regressão Linear Simples para prever um valor contínuo: a progressão da doença a partir do IMC.")
print(f"O R2 obtido foi de {r2:.2f}, indicando que o modelo explica aproximadamente 23% da variação.")
print("A diferença relativamente ao exercício Iris depende dos dados, das variáveis utilizadas e da metodologia aplicada. O Iris é originalmente um dataset de classificação.")
