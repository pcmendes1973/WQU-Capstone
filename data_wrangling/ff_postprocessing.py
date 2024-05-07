import pandas as pd
import numpy as np
import sys
from utils.utils import load_config


def compile_ff_data(price_data, root_dir):
    tickers = price_data.columns
    master_df = pd.DataFrame()

    for tic in tickers:
        print(f'Processing {tic}...')
        income_statement = pd.read_csv(f'{root_dir}/IncomeStatement/{tic}_IS.csv')
        balance_sheet = pd.read_csv(f'{root_dir}/BalanceSheet/{tic}_BS.csv')
        cash_flow = pd.read_csv(f'{root_dir}/CashFlow/{tic}_CF.csv')
        df = pd.merge(income_statement, balance_sheet, on='fiscalDateEnding', how='outer').merge(cash_flow, on='fiscalDateEnding', how='outer')
        df = df.rename(columns={'fiscalDateEnding': 'Date'})
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.resample('QE').last() # resample to quarters so datasets all match

        data_items_needed = ['dividendPayout', 'commonStockSharesOutstanding', 'netIncome_x', 'operatingCashflow', 'totalRevenue', 'costOfRevenue', 'totalAssets']

        subset_df = df[data_items_needed]
        subset_df = subset_df.ffill().bfill() # fill in missing quarterly data
        subset_df = subset_df.resample('ME').ffill() # resample to monthly before joining with price data
        price_data_m = price_data.resample('ME').last() # resample price_data to monthly


        subset_df = pd.merge(subset_df, price_data_m[[tic]], on='Date')
        subset_df = subset_df.rename(columns={tic: 'Price'})

        # calculate factors
        subset_df['dividendYield'] = subset_df.dividendPayout / subset_df.commonStockSharesOutstanding / subset_df.Price
        subset_df['payoutRatio'] = subset_df.dividendPayout / subset_df.netIncome_x
        subset_df['operatingcashFlowRatio'] = subset_df.operatingCashflow / (subset_df.totalRevenue - subset_df.costOfRevenue)
        subset_df['ROA'] = subset_df.netIncome_x / subset_df.totalAssets
        subset_df['netProfitMargin'] = subset_df.netIncome_x / subset_df.totalRevenue
        subset_df['ticker'] = tic

        # replace any divide by 0 errors
        subset_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        subset_df.ffill(inplace=True)

        subset_df.drop(['Price'] + data_items_needed, axis=1, inplace=True)

        master_df = pd.concat([master_df, subset_df], axis=0)
        master_df = master_df.sort_values(by=['Date', 'ticker'])

    return master_df


def main():
    price_data = pd.read_csv('../data/price_data.csv', index_col='Date')
    price_data.index = pd.to_datetime(price_data.index)

    root_dir = '../data/statement_data/'
    config = load_config()
    start_date = config.get('MetaData', 'final_start_date')

    compiled_df = compile_ff_data(price_data, root_dir)

    fundamental_df = compiled_df.rename(columns={'ticker': 'Ticker'})
    fundamental_df.set_index(['Ticker'], append=True, inplace=True) # create multi-index with Date
    fundamental_df = fundamental_df.loc[start_date:] # set start date

    fundamental_df.to_csv('../data/monthly_fund_data.csv')

    print('\n\nFundamental data compilation is complete!\n')


if __name__=="__main__":
    sys.exit(main())