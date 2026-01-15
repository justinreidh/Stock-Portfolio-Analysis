import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import streamlit as st
import os
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

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

@st.cache_data(ttl=3600)  
def load_stock_data(tickers_list, start, end):
    data = yf.download(tickers_list, start=start, end=end, auto_adjust=False, progress=False)
    prices = data.xs('Adj Close', level=0, axis=1, drop_level=True)
    
    sector_map = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info
            sector = info.get('sector', 'Unknown')
            if sector in ['N/A', '', None]:
                sector = 'Unknown'
            sector_map[t] = sector
        except:
            sector_map[t] = 'Unknown'
    
    meta_df = pd.DataFrame({
        'ticker': list(sector_map.keys()),
        'sector': list(sector_map.values())
    })
    
    return prices, meta_df, sector_map

prices, meta_df, sector_map = load_stock_data(tickers, start_date, end_date)

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

# Calculate rolling correlations 

window_short = 30   
window_long = 252   

rolling_corr_short = daily_returns.rolling(window_short).corr().dropna()
rolling_corr_long  = daily_returns.rolling(window_long).corr().dropna()

latest_rolling_short = rolling_corr_short.groupby(level=0).last()  

print("Latest 30-day rolling correlation matrix (sample):\n")
print(latest_rolling_short.tail(10))  


# Summarize and explore data

print("Latest Cumulative Returns by Sector (%):\n")
print(latest_sector_returns.sort_values(ascending=False))

summary = pd.DataFrame({
    'Mean Daily Return (%)': daily_returns.mean() * 100,
    'Volatility (Std Dev)': daily_returns.std(),
    'Total Return (%)': cumulative_returns.iloc[-1] * 100
})



sector_correlations = sector_daily_returns.corr()

avg_correlation = sector_correlations.mean().sort_values()

heatmap_path = "sector_correlation_heatmap.png"
if not os.path.exists(heatmap_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sector_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Sector Return Correlations')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close() 

# Heirarchical Cluster and Dendrogram

corr_matrix = daily_returns.droplevel('sector', axis=1).corr()

@st.cache_data
def get_clustering_data(corr):
    dist = 1 - corr.abs()
    condensed = squareform(dist)
    link = linkage(condensed, method='ward')
    return link, corr.columns

linkage_matrix, labels = get_clustering_data(corr_matrix)

# Then display dendrogram as static image (like before)
dendro_path = "stock_dendrogram.png"
if not os.path.exists(dendro_path):
    plt.figure(figsize=(12,8))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, color_threshold=0.7)
    plt.title('Stock Clustering Dendrogram')
    plt.xlabel('Stocks')
    plt.ylabel('Distance (1 - |Correlation|)')
    plt.subplots_adjust(bottom=0.25) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    plt.savefig(dendro_path, dpi=150, bbox_inches='tight')  # bbox_inches='tight' crops whitespace intelligently
    plt.close()


distance_threshold = 0.7

clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

cluster_df = pd.DataFrame({
    'Stock': corr_matrix.columns,
    'Cluster': clusters
}).sort_values('Cluster')

# Try a simple linear regression

aapl_prices = prices['AAPL'].dropna()
X = np.arange(len(aapl_prices)).reshape(-1, 1)  
y = aapl_prices.values

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(aapl_prices), len(aapl_prices) + 30).reshape(-1, 1)
predictions = model.predict(future_days)

st.title('Stock Analyzer')

selected_ticker = st.selectbox('Select Individual Stock', tickers)
st.line_chart(prices[selected_ticker])

st.dataframe(summary)

st.subheader("Sector Performance Overview")
st.dataframe(latest_sector_returns.rename("Cumulative Return (%)").sort_values(ascending=False))

st.subheader("Sector Correlation Heatmap")
st.image(heatmap_path, width="stretch")

st.subheader("Most Diversifying Sectors")
st.write("Sectors with lowest average correlation (better for diversification):")
st.dataframe(avg_correlation.rename("Avg Correlation").round(2))

selected_sector = st.selectbox("Filter by Sector", options=['Select a Sector'] + sorted(meta_df['sector'].unique()))
if selected_sector != 'Select a Sector':
    filtered_tickers = meta_df[meta_df['sector'] == selected_sector]['ticker'].tolist()
    if filtered_tickers: 
        st.line_chart(prices[filtered_tickers])
    else:
        st.warning(f"No stocks found in sector: {selected_sector}")

st.subheader("Cluster-Based Diversification Suggestions")
st.write("Pick **one stock from each cluster** to build a more diversified portfolio:")

st.image(dendro_path, width="stretch")

for cluster_id, group in cluster_df.groupby('Cluster'):
    st.write(f"**Cluster {cluster_id}** ({len(group)} stocks - highly similar):")
    st.write(", ".join(group['Stock'].tolist()))

# python -m streamlit run stock_analyzer.py