# Regressão Linear Simples — Dataset Diabetes

Projeto prático de Machine Learning desenvolvido em Python com o objetivo de prever a progressão da diabetes um ano após o início do estudo a partir do Índice de Massa Corporal (IMC), utilizando o algoritmo de **Regressão Linear Simples**.

## 📊 Dataset Utilizado
* **Fonte:** `sklearn.datasets.load_diabetes(scaled=False)` (scikit-learn)
* **Amostras:** 442 doentes diabéticos
* **Variável Independente ($X$):** Índice de Massa Corporal — IMC ($\text{kg/m}^2$)
* **Variável Dependente ($y$):** Medida quantitativa da progressão da doença (valores entre 25 e 346)

## 🛠️ Tecnologias e Bibliotecas
* **Python 3**
* **scikit-learn:** carregamento de dados, divisão treino/teste (`80/20`), treino do modelo (`LinearRegression`) e cálculo de métricas ($R^2$ e $MAE$)
* **matplotlib:** visualização da dispersão dos dados reais e da reta de regressão
* **numpy:** análise de resíduos e identificação de casos extremos

## 📈 Resultados Obtidos
* **Equação da reta:** $\hat{y} \approx -127.82 + 10.82x$
* **Coeficiente de Determinação ($R^2$):** $\approx 0.23$
* **Erro Médio Absoluto ($MAE$):** $\approx 52$ pontos
* **Previsão para $IMC = 30$:** $\approx 191$ pontos

## 🚀 Como Executar
1. Clone este repositório:
   ```bash
   git clone [https://github.com/SEU-UTILIZADOR/regressao-linear-simples-diabetes.git](https://github.com/SEU-UTILIZADOR/regressao-linear-simples-diabetes.git)
