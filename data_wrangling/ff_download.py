import pandas as pd
import requests
import sys

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
def get_alpha_vantage_data(statement, ticker, token):
  url = f'https://www.alphavantage.co/query?function={statement}&symbol={ticker}&apikey={token}'
  r = requests.get(url)
  return r.json()

def get_fundamental_factors(ticker, token):
  dfs = []
  for statement in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']:
    data = get_alpha_vantage_data(statement, ticker, token)
    df = pd.DataFrame(data['quarterlyReports'])
    df = df.drop('reportedCurrency', axis=1)
    dfs.append(dfs)
  return pd.concat(dfs, axis=1)

def main(tickers):
  # get Alpha Vantage api key
  with open("../alpha_vantage_api_key.txt", 'r') as f:
    AV_token = f.read()

  master_df = pd.DataFrame()

  for ticker in tickers:
    statement_df = get_fundamental_factors(ticker, AV_token)
    
    master_df = pd.concat([master_df, statement_df], axis=0)

    return master_df
  
if __name__=="__main__":
  sys.exit(main())
    