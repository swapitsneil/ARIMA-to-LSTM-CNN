# Neural Network Python Tutorial: From ARIMA to LSTM-CNN Hybrid

## 📌 Objective
This project demonstrates how to forecast stock prices using:
1. **ARIMA** – A classic statistical time series model.
2. **LSTM-CNN Hybrid** – A modern deep learning model combining Convolutional Neural Networks (CNN) for local pattern extraction and Long Short-Term Memory (LSTM) for long-term dependencies.

The workflow starts with an ARIMA baseline and transitions to a hybrid neural network to capture complex patterns in financial time series data.  
We also implement **rolling forecasting** for realistic multi-step predictions.

---

## 📂 Dataset
- **Source:** [Yahoo Finance](https://finance.yahoo.com/)
- **Ticker:** `AAPL` (Apple Inc.)
- **Period:** 2020-01-01 → 2024-01-01
- **Feature Used:** Closing Price (`Close` column)

---

## ⚙️ Installation

Clone the repository:

    git clone https://github.com/yourusername/ARIMA-to-LSTM-CNN.git
    cd ARIMA-to-LSTM-CNN

Install dependencies:

    pandas
    numpy
    matplotlib
    yfinance
    statsmodels
    scikit-learn
    tensorflow

### 🚀 Usage

Run the script:

    python forecast.py

### Output:

Plots comparing:

  - Actual stock prices
  - ARIMA forecast
  - LSTM-CNN one-step predictions
  - LSTM-CNN rolling forecast predictions

  ![Image](https://github.com/user-attachments/assets/835d3bab-31b0-4eca-ac7e-44f7ba33a63f)

## 🧠 Code Structure

### 1. Importing Libraries
  Handles data manipulation, visualization, statistical modeling, scaling, and deep learning model creation.


### 2. Loading Data
  Fetches Apple stock data and extracts the Close column. 
  
    ticker = 'AAPL'
    data = yf.download(ticker, start='2020-01-01', end='2025-01-01')
    close_prices = data['Close']
  

### 3. ARIMA Baseline
  Splits the data, fits an ARIMA model, and generates predictions.

    train_size = int(len(close_prices) * 0.8)
    train_arima, test_arima = close_prices[:train_size], close_prices[train_size:]
    arima_model = ARIMA(train_arima, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_preds = arima_fit.forecast(len(test_arima))


### 4. Data Scaling & Sequence Creation
  Scales prices to [0,1] range and converts time series into sequences for neural networks.


### 5. Building the LSTM-CNN Model

### 6. Results Visualization
  Plots actual vs predicted prices for ARIMA and LSTM-CNN models.

  Model results

  ![Image](https://github.com/user-attachments/assets/62716bc3-7384-4907-872f-d7c99f3190b4)


## 📊 Key Insights
  ARIMA works well for simpler patterns but struggles with non-linearity.
  LSTM-CNN Hybrid learns both short-term and long-term dependencies.
  Rolling Forecasting shows real-world performance without relying on actual future values.

### Contact - for queries and issues
   email - swapnilnicolson.201@gmail.com


    
