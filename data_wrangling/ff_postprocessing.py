import pandas as pd
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
        df = df.resample('Q').last() # resample to quarters so datasets all match

        data_items_needed = ['dividendPayout', 'commonStockSharesOutstanding', 'netIncome_x', 'operatingCashflow', 'totalRevenue', 'costOfRevenue', 'totalAssets']

        subset_df = df[data_items_needed]
        subset_df = pd.merge(subset_df, price_data[[tic]], on='Date')
        subset_df = subset_df.rename(columns={tic: 'Price'})

        # calculate factors
        subset_df['dividendYield'] = subset_df.dividendPayout / subset_df.commonStockSharesOutstanding / subset_df.Price
        subset_df['payoutRatio'] = subset_df.dividendPayout / subset_df.netIncome_x
        subset_df['operatingcashFlowRatio'] = subset_df.operatingCashflow / (subset_df.totalRevenue - subset_df.costOfRevenue)
        subset_df['ROA'] = subset_df.netIncome_x / subset_df.totalAssets
        subset_df['netProfitMargin'] = subset_df.netIncome_x / subset_df.totalRevenue
        subset_df['ticker'] = tic

        subset_df.drop(['Price'] + data_items_needed, axis=1, inplace=True)

        master_df = pd.concat([master_df, subset_df], axis=0)
        master_df = master_df.sort_values(by=['Date', 'ticker'])

    return master_df


def main():
    price_data = pd.read_csv('../data/price_data.csv', index_col='Date')
    price_data.index = pd.to_datetime(price_data.index)

    root_dir = '../data/statement_data/'
    # start_date = '2010-06-30' # might be nice to get into a config file
    config = load_config()
    start_date = config.get('MetaData', 'start_date')

    compiled_df = compile_ff_data(price_data, root_dir)

    fundamental_df = compiled_df.rename(columns={'ticker': 'Ticker'})

    # Set start date and handle missing values
    fundamental_df.set_index(['Ticker'], append=True, inplace=True)
    fundamental_df = fundamental_df.loc[start_date:]
    fundamental_df = fundamental_df.ffill().bfill()

    # Resample quarterly data to monthly
    monthly_ff = fundamental_df.reset_index(level=['Ticker'])
    monthly_ff = monthly_ff.groupby(['Ticker']).resample('M').ffill()
    monthly_ff.drop('Ticker', axis=1, inplace=True)


    monthly_ff.to_csv('../data/monthly_fund_data.csv')



if __name__=="__main__":
    sys.exit(main())