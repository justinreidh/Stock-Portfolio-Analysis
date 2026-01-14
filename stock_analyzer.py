import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import streamlit as st

sns.set_style("whitegrid")

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
start_date = '2024-01-01'  
end_date = datetime.today().strftime('%Y-%m-%d') 

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)

prices = data.xs('Adj Close', level=0, axis=1, drop_level=True)

print(prices.head())

# Clean the Data

prices = prices.fillna(method='ffill')

daily_returns = prices.pct_change().dropna()

monthly_returns = daily_returns.resample('M').sum() * 100  

cumulative_returns = (1 + daily_returns).cumprod() - 1

rolling_avg = prices.rolling(window=50).mean()

summary = pd.DataFrame({
    'Mean Daily Return (%)': daily_returns.mean() * 100,
    'Volatility (Std Dev)': daily_returns.std(),
    'Total Return (%)': cumulative_returns.iloc[-1] * 100
})

print("Monthly Returns Sample:\n", monthly_returns.head())
print("\nPortfolio Summary:\n", summary)

plt.figure(figsize=(12, 6))
prices.plot()
plt.title('Stock Prices Over Time')
plt.ylabel('Adjusted Close Price')
plt.savefig('prices_plot.png')  
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm')
plt.title('Stock Return Correlations')
plt.savefig('correlation_heatmap.png')
plt.show()

summary['Total Return (%)'].plot(kind='bar', figsize=(10, 5))
plt.title('Total Portfolio Returns')
plt.ylabel('Return (%)')
plt.savefig('total_returns_bar.png')
plt.show()

aapl_prices = prices['AAPL'].dropna()
X = np.arange(len(aapl_prices)).reshape(-1, 1)  
y = aapl_prices.values

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(aapl_prices), len(aapl_prices) + 30).reshape(-1, 1)
predictions = model.predict(future_days)

plt.figure(figsize=(12, 6))
plt.plot(aapl_prices.index, aapl_prices, label='Historical')
plt.plot(pd.date_range(aapl_prices.index[-1], periods=31)[1:], predictions, label='Forecast')
plt.title('AAPL Price Forecast')
plt.legend()
plt.savefig('forecast_plot.png')
plt.show()

st.title('Personal Stock Portfolio Analyzer')
selected_ticker = st.selectbox('Select Stock', tickers)
st.line_chart(prices[selected_ticker])
st.dataframe(summary)