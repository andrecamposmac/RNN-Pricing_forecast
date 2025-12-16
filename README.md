# Previsão de Vendas Mensais com RNN (LSTM)

Este projeto implementa uma Rede Neural Recorrente (RNN), especificamente uma **LSTM (Long Short-Term Memory)**, para realizar a previsão de vendas mensais no varejo. O objetivo é prever os valores de vendas para os próximos 12 meses com base no histórico temporal.

## Descrição

O notebook aborda o problema de *Time Series Forecasting* (Previsão de Séries Temporais) utilizando Deep Learning. O pipeline do projeto inclui:
1.  **Coleta e Carregamento de Dados:** Leitura de dados históricos de vendas.
2.  **Pré-processamento:**
    *   Conversão da coluna de data para índice.
    *   Normalização dos dados utilizando `MinMaxScaler` para escalar os valores entre 0 e 1 (crucial para redes neurais).
    *   Criação de sequências temporais (lags) utilizando `TimeseriesGenerator`.
3.  **Modelagem:** Construção e treinamento de uma rede LSTM.
4.  **Validação:** Teste do modelo em dados não vistos (conjunto de teste) e uso de *Early Stopping* para evitar overfitting.
5.  **Previsão Futura:** Geração de previsões para 12 meses além do dataset original.

## Dataset

Os dados utilizados representam as **Vendas Mensais Antecipadas para Varejo e Serviços de Alimentação** (Advance Monthly Sales for Retail and Food Services).

*   **Fonte:** [FRED - Federal Reserve Economic Data](https://fred.stlouisfed.org/series/MRTSSM448USN)
*   **Unidade:** Milhões de Dólares (Não ajustados sazonalmente).
*   **Frequência:** Mensal.
*   **Período:** Dados desde 1992.

## Tecnologias e Bibliotecas

*   **Python 3**
*   **TensorFlow / Keras:** Para criação e treinamento da rede neural (LSTM).
*   **Pandas & NumPy:** Manipulação de séries temporais e arrays.
*   **Scikit-learn:** Normalização dos dados (`MinMaxScaler`).
*   **Matplotlib:** Visualização de gráficos de vendas e previsões.

## Arquitetura do Modelo

A rede neural foi configurada da seguinte forma:
*   **Entrada:** Sequências temporais geradas pelo `TimeseriesGenerator`.
*   **Camada LSTM:** 100 unidades (neurônios).
*   **Camada Densa (Saída):** 1 unidade (valor previsto).
*   **Otimizador:** Adam (padrão do Keras).
*   **Loss Function:** MSE (Mean Squared Error).
*   **Callback:** EarlyStopping monitorando `val_loss` com paciência de 2 épocas.

## Resultados

O modelo demonstrou capacidade de capturar a sazonalidade e tendência dos dados de vendas.
*   O notebook compara as **Vendas Reais vs. Previsões** no conjunto de teste.
*   Ao final, é gerada uma tabela com o **Forecast** (previsão futura) para o período de **Novembro de 2019 a Outubro de 2020**.

## Como Executar

1.  Certifique-se de ter o arquivo `sales_price.csv` ou atualize o código para baixar diretamente da fonte.
2.  Abra o notebook `Pricing_forecast_RNN.ipynb` no Google Colab ou Jupyter.
3.  Ajuste o caminho do arquivo na célula de carregamento:
    ```python
    df = pd.read_csv('caminho/para/sales_price.csv', ...)
    ```
4.  Execute todas as células para treinar o modelo e visualizar as previsões.

---
