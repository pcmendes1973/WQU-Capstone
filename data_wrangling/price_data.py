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
from utils.utils import load_config
import os
import sys

# SP 500 dividend aristocrats as of April 2024
tickers = ['ABBV', 'BDX', 'CL', 'FAST', 'KO', 'NUE', 'SWK',
           'ABT', 'BEN', 'CLX', 'FRT', 'KVUE', 'O', 'SYY',
           'ADM', 'BF-B', 'CTAS', 'GD', 'LIN', 'PEP', 'TGT',
           'ADP', 'BRO', 'CVX', 'GPC', 'LOW', 'PG', 'TROW',
           'AFL', 'CAH', 'DOV', 'GWW', 'MCD', 'PNR', 'WMT',
           'ALB', 'CAT', 'ECL', 'HRL', 'MDT', 'PPG', 'WST',
           'AMCR', 'CB', 'ED', 'IBM', 'MKC', 'ROP', 'XOM',
           'AOS', 'CHD', 'EMR', 'ITW', 'MMM', 'SHW',
           'APD', 'CHRW', 'ESS', 'JNJ', 'NDSN', 'SJM',
           'ATO', 'CINF', 'EXPD', 'KMB', 'NEE', 'SPGI']

def main():

    config = load_config()
    start_date = config.get('MetaData', 'raw_start_date')
    end_date = config.get('MetaData', 'end_date')

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

    price_data.to_csv('../data/price_data.csv')

    print('\n\nPrice data download is complete!\n')

if __name__=="__main__":
    sys.exit(main())
