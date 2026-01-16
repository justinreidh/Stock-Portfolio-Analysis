# Stock Market Analyzer Dashboard

Interactive web dashboard for analyzing stocks using financial metrics, sector performance, correlations, hierarchical clustering, and basic portfolio optimization.

Built with **Streamlit**, **yfinance**, **pandas**, **matplotlib/seaborn**, **scikit-learn**, and **scipy**.


## Main Features

- Daily returns, cumulative returns, and sector-level performance
- Sector comparison (latest cumulative return %)
- Sector correlation heatmap
- Hierarchical clustering of stocks (dendrogram based on correlation distance)
- 50-day rolling average price charts
- Interactive individual stock price charts
- Monte Carlo simulation of 10,000 random portfolios
- Optimized portfolios:
  - Minimum Volatility portfolio
  - Maximum Sharpe Ratio portfolio (risk-free rate = 2%)
- Ranking of most diversifying sectors (lowest average correlation)

## Current Limitations

- Fixed hardcoded list of 25 tickers (no dynamic watchlist yet)
- Very basic linear regression trend example (not displayed in UI)
- Missing prices are forward-filled (may distort returns near gaps)
- Portfolio optimization assumptions:
  - Long-only (weights between 0–1)
  - No transaction costs, taxes, or slippage
  - Annualization factor of 252 trading days
  - Fixed 2% risk-free rate
- Only 10,000 Monte Carlo samples
- No advanced techniques (CVaR, Black-Litterman, constraints, factor models, etc.)
- No backtesting or rebalancing simulation

## Possible Future Enhancements

- User-defined ticker list 
- Interactive date range picker in the UI
- More advanced portfolio optimization 
- Factor exposure analysis / risk parity
- Portfolio strategy backtesting

## Requirements

- **Python** ≥ 3.10  
  (Developed and tested on Python 3.13.5 – Windows 10 / bash)

### Recommended installation

```bash
pip install --upgrade streamlit yfinance pandas matplotlib seaborn scikit-learn scipy numpy
