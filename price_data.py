# -*- coding: utf-8 -*-
"""
Downloads prices for the S&P dividend aristocrats from 2000 to 2024
from Yahoo!Finance and treats the data.

The following operations are performed:
    - Remove tickers with insufficient data (3 total)
    - Saves the data into a dataframe
    - Saves the dataframe into a .csv file

Created on Sun Apr 28 12:31:34 2024

@author: pcmen
"""

import pandas as pd
import yfinance as yf

sp_da = pd.read_html('https://en.wikipedia.org/wiki/S%26P_500_Dividend_Aristocrats')[0]
tickers = sp_da['Ticker symbol'].tolist()

start_date = "2000-01-01"
end_date = "2024-04-07"

try:
    price_data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True)['Close']
except Exception as e:
    print(f"Exception: Args {e.args}")

price_data = price_data.loc[:, price_data.iloc[-1].notnull() & price_data.iloc[0].notnull()]
tickers = list(price_data.columns)

price_data.to_csv('Price_data.csv')