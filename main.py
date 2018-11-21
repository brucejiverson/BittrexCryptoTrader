# secret sandbox api key gSrdc5nqcjRKnpZM0/c6VbVI889iBVpSmzYGJZd3m4Jj1Uc1ZN+N95Yi+ytYZwe9tnzbt5A6BUulkf2zE13X1A==
# sandbox api key 9d8ad1d74c1f2beb08ba3577bd791ee2

import my_gdax
from datetime import datetime, timedelta
import cbpro
from artemis import strategy
import pandas as pd

def ROI(final, initial):
    return round(final / initial - 1, 5) * 100

def backtest(start, end, desired_trading_period):
    #desired trading period in
    granularity = 60

    account_log = pd.DataFrame(
        columns=['Date', 'Account Value (USD)', 'Trades'])
    data = my_gdax.my_gdax("BTC-USD", 'backtest')
    data.authenticated()
    data.initialize(start, desired_trading_period)
    data.process()
    in_market = False
    money = 100
    fees = 0.003  # standard fees are 0.3% per transfer
    data.plot_data(ROI(100, 100), ROI(data.df.tail(1)['close'], data.df.loc[0, 'close']))
    # for each data point from start to finish, check the strategy and calculate the money
    date = start
    # while date <= end:
    #     # get the next data point and append it to the data frame
    #     date = min(start + timedelta(seconds = data.granularity), end)
    #     data.new_datapoint(date)
    #
    #     type = ''
    #     strategy_result = strategy(data, desired_trading_period)
    #     if (strategy_result == "bullish" and not in_market):
    #         # buy
    #         # log trade
    #         money *= slice.loc['close']
    #         type = "buy"
    #     elif (strategy_result == "bearish" and in_market):
    #         # sell
    #         # log trade
    #         money = money / slice.loc['close']
    #         type = "sell"
    #         pass
    #
    #     # determine the account value
    #     if in_market:
    #         account_value = money / slice.loc['close']
    #     else:
    #         account_value = money
    #     # update log w date from row, account value, and if a trade was executed
    #     account_log.append(pd.DataFrame(data=[slice.loc['date'], account_value, type], columns=[
    #                        'Date', 'Account Value (USD)', 'Trades']), ignore_index=True)

    # plot the account value

    #return ROI(account_value, 100), account_log


def run():
    pass


start_date = datetime(2018, 10, 26)
end_date = datetime(2018, 10, 27)
backtest(start_date, end_date, 3)
