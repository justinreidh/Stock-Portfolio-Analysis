import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import streamlit as st

sns.set_style("whitegrid")

tickers = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META',     # Tech
    'JPM', 'V', 'MA', 'BRK-B',                           # Financials
    'TSLA', 'HD', 'MCD',                                 # Consumer Discretionary
    'PG', 'KO', 'WMT',                                   # Consumer Staples
    'LLY', 'JNJ', 'PFE',                                 # Health Care
    'XOM', 'CVX',                                        # Energy
    'BA', 'CAT',                                         # Industrials
    'NFLX',                                              # Communication Services
    'NEE'                                                # Utilities
]
start_date = '2024-01-01'  
end_date = datetime.today().strftime('%Y-%m-%d') 

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)

prices = data.xs('Adj Close', level=0, axis=1, drop_level=True)

print(prices.head())

# Fetch metadata for each tickers

sector_map = {}
for t in tickers:
    try:
        info = yf.Ticker(t).info
        sector = info.get('sector', 'Unknown')
        if sector in ['N/A', '', None]:
            sector = 'Unknown'
        sector_map[t] = sector
    except Exception:
        sector_map[t] = 'Unknown'

meta_df = pd.DataFrame({
    'ticker': list(sector_map.keys()),
    'sector': list(sector_map.values())
})

print("Sector distribution:\n", meta_df['sector'].value_counts())

# Clean the Data

prices = prices.fillna(method='ffill')

daily_returns = prices.pct_change().dropna()
daily_returns.columns = pd.MultiIndex.from_tuples(
    [(col, sector_map.get(col, 'Unknown')) for col in daily_returns.columns],
    names=['ticker', 'sector']
)
sector_daily_returns = daily_returns.groupby(level='sector', axis=1).mean()

monthly_returns = daily_returns.resample('M').sum() * 100  

cumulative_returns = (1 + daily_returns).cumprod() - 1
sector_cumulative_returns = (1 + sector_daily_returns).cumprod() - 1

latest_sector_returns = sector_cumulative_returns.iloc[-1] * 100

rolling_avg = prices.rolling(window=50).mean()

# Summarize and explore data

print("Latest Cumulative Returns by Sector (%):\n")
print(latest_sector_returns.sort_values(ascending=False))

summary = pd.DataFrame({
    'Mean Daily Return (%)': daily_returns.mean() * 100,
    'Volatility (Std Dev)': daily_returns.std(),
    'Total Return (%)': cumulative_returns.iloc[-1] * 100
})

print("\nPortfolio Summary:\n", summary)

'''

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

'''

# Sector cumulative returns plot
plt.figure(figsize=(12, 6))
sector_cumulative_returns.plot()
plt.title('Cumulative Returns by Sector')
plt.ylabel('Cumulative Return')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('sector_cumulative_returns.png')
plt.show()

sector_correlations = sector_daily_returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(sector_correlations, annot=True, cmap='coolwarm')
plt.title('Sector Return Correlations')
plt.savefig('sector_correlation_heatmap.png')
plt.show()

avg_correlation = sector_correlations.mean().sort_values()

print("\nSectors ranked by average correlation (lowest = most diversifying):\n")
print(avg_correlation.round(2))

# Try a simple linear regression

aapl_prices = prices['AAPL'].dropna()
X = np.arange(len(aapl_prices)).reshape(-1, 1)  
y = aapl_prices.values

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(aapl_prices), len(aapl_prices) + 30).reshape(-1, 1)
predictions = model.predict(future_days)

'''
plt.figure(figsize=(12, 6))
plt.plot(aapl_prices.index, aapl_prices, label='Historical')
plt.plot(pd.date_range(aapl_prices.index[-1], periods=31)[1:], predictions, label='Forecast')
plt.title('AAPL Price Forecast')
plt.legend()
plt.show()
'''

st.title('Stock Analyzer')
selected_ticker = st.selectbox('Select Stock', tickers)
st.line_chart(prices[selected_ticker])
st.dataframe(summary)

st.subheader("Sector Overview")
st.dataframe(meta_df[['ticker', 'sector']])

selected_sector = st.selectbox("Filter by Sector", options=['All'] + list(meta_df['sector'].unique()))
if selected_sector != 'All':
    filtered_tickers = meta_df[meta_df['sector'] == selected_sector]['ticker'].tolist()
    st.line_chart(prices[filtered_tickers])

st.subheader("Sector Performance Overview")
st.dataframe(latest_sector_returns.rename("Cumulative Return (%)").sort_values(ascending=False))

st.subheader("Sector Correlation Heatmap")
st.pyplot(plt.gcf()) 

st.subheader("Most Diversifying Sectors")
st.write("Sectors with lowest average correlation (better for diversification):")
st.dataframe(avg_correlation.rename("Avg Correlation").round(2))