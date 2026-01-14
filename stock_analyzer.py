import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

sns.set_style("whitegrid")

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
start_date = '2024-01-01'  
end_date = datetime.today().strftime('%Y-%m-%d') 

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)

prices = data.xs('Adj Close', level=0, axis=1, drop_level=True)

print(prices.head())