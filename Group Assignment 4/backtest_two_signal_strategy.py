'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 6 May 2022

This program fetches daily close prices from the internet for given tickers and time frame,
and then back tests some simple momentum and reversal monthly strategies in conjuction to
figure out the most optimum coefficients for their weightage under a linear regression,
building on the existing predictions

The program can be called using:
backtest_two_signal_strategy.py --tickers <ttt> –-b <YYYYMMDD> –-e <YYYYMMMDD>
--initial_aum <yyyy>  --strategy1_type <xx> --days1 <xx> ---strategy2_type <xx>
--days2 <xx> --top_pct <xx>

A sample means for calling the function is:
python backtest_two_signal_strategy.py
--ticker 'AAPL,TSLA,MSFT,FB,WMT,PFE,SPY,AMZN,BXP,DLR,QQQ,VUG,IWF,IJH'
--b 20210112 --e 20220503 --initial_aum 1000 --strategy1_type 'M'
--days1 50 --strategy2_type 'M' --days2 100 --top_pct 25

'''
import argparse
import os
from datetime import date, timedelta
from math import ceil
from classes import Ticker, Portfolio
import warnings
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
warnings.filterwarnings('ignore')

def main(tickers,b_date,e_date,initial_aum,strategy1_type,days1,strategy2_type,days2,top_pct):
    '''
    This is the main function for initializing, computing, and retrieving the final output.
    The printed outputs are also generated here.

    Inputs:
    tickers (string): a list of 4 character ticker name strings, separated by commas
    b_date (string): the date the initial AUM was invested
    e_date (string): the final date to be used for calculations. If no ending date is
    provided by the user, the script takes the latest possible date
    initial_aum (integer): the intial amount of assets that are invested in the portfolio
    strategy_type (string): the backtesting strategy chosen by the user
    days (int): the numbers of trading days used to compute strategy-related returns
    top_pct (int): a positive integer from 1 to 100 that indicates the percentage of stocks
    to pick to go long

    Returns:
    portfolio (class): the module containing all the relevant information about the portfolio
    '''
    if days1 > 250 or days2 > 250:
        raise ValueError('--days exceeds 250 days.')
    if days1 < 1 or days2 < 1:
        raise ValueError('--days is not a positive number.')
    if top_pct < 1 or top_pct > 100:
        raise ValueError('The integer for the percentage of stocks needs to be between 1 and 100.')
    if strategy1_type not in ['M','R'] or strategy2_type not in ['R','M']:
        raise ValueError('There is no such strategy type.')
    if initial_aum <= 0:
        raise ValueError('The initial asset under management is not a positive value')
    try:
        beginning_date=date(int(str(b_date)[:4]), int(str(b_date)[4:6]), int(str(b_date)[6:]))
        if e_date is None:
            ending_date=date.today()
        else:
            ending_date=date(int(str(e_date)[:4]), int(str(e_date)[4:6]), int(str(e_date)[6:]))
            if (ending_date-beginning_date).days<0:
                raise ValueError('The ending date is before the beginning date')
            if (ending_date-date.today()).days>0:
                raise ValueError("The ending date is after today's date. No data available.")
    except ValueError as v_e:
        raise ValueError('You have entered an invalid date.') from v_e

    tickers_list = tickers.split(',')
    for ticker in tickers_list:
        print('Verifying %s...' % ticker)
        ticker_data = yf.Ticker(ticker).history(period='max')
        if len(ticker_data.index) == 0:
            raise NameError("No stock ticker found for %s." % ticker)
        if (beginning_date - ticker_data.index[0].date()).days < 450:
            raise ValueError('Ticker %s does not have sufficient data (less than 250 trading days\
            prior to beginning date' % ticker)
        ticker_data = yf.Ticker(ticker).history(start = beginning_date+timedelta(1), \
        end = ending_date + timedelta(1))
        delta = 0
        final_ind = 0
        for ind, cur_date in enumerate(ticker_data.index):
            delta = cur_date.month - beginning_date.month
            if delta != 0:
                final_ind = ind-1
                break
        if final_ind <= 0:
            raise ValueError('Not enough data for backtesting.')

    print("Final touches...")
    portfolio = Portfolio(tickers_list, beginning_date, ending_date,
    initial_aum,strategy1_type, days1, strategy2_type, days2, top_pct)
    beginning_date = portfolio.beginning_date
    ending_date = portfolio.ending_date
    num_days = portfolio.calc_num_of_days(beginning_date, ending_date)
    total_return = portfolio.calc_tr(beginning_date, ending_date)
    aror= portfolio.calc_aror(total_return)
    final_aum = portfolio.get_final_value(ending_date)
    average_aum = portfolio.calc_avg_aum()
    max_aum = portfolio.get_max_aum()
    pnl = portfolio.get_pnl(beginning_date,ending_date)
    adr = portfolio.get_adr(total_return)
    std = portfolio.get_std()
    sharpe = portfolio.get_sharpe(adr, std)

    numbers = ["(1)", "(2)", "(3)", "(4)"]
    prompts = ["Begin date:  ", "End date:  ", "Number of days:  ", "Total Stock Return:  "]
    responses = [beginning_date, ending_date, num_days, '']
    for tick in tickers_list:
        prompts.append(tick+":  ")
        numbers.append('')
        total_stock_return = portfolio.calc_tsr(tick)
        if total_stock_return != 0:
            responses.append(round(total_stock_return, 5))
        else:
            responses.append('This stock was not purchased in the strategy.')
    prompts += ["Total Return:  ", "Annualized RoR:  ", "Initial AUM:  ", "Final AUM:  ",
    "Average AUM:  ", "Maximum AUM:  ", "PnL of AUM:  ", "Average Daily Return:  ",
    "SD of Daily Return:  ", "Daily Sharpe Ratio:  ", "Linear Reg Models:  "]
    responses += [round(total_return,5), round(aror,5), round(initial_aum,2), round(final_aum,2),
    round(average_aum,2), round(max_aum,2), round(pnl,5), round(adr,5), round(std,5),
    round(sharpe,5),
    'Strategy 1 (%s-%d)  ||  Strategy 2 (%s-%d)' % (strategy1_type, days1, strategy2_type, days2)]
    numbers += ["(5)", "(6)", "(7)", "(8)", "(9)", "(10)", "(11)", "(12)", "(13)", "(14)", "(15)"]
    coefs = portfolio.tick_data.coefs
    prompts += ['', "Coeff:  ", "T-Val:  "]
    numbers += ['', '', '']
    coef1 = '{: f}'.format(round(coefs[0], 5))
    coef2 = '{: f}'.format(round(coefs[1], 5))
    tval1 = '{: f}'.format(round(portfolio.tick_data.tvals[1], 5))
    tval2 = '{: f}'.format(round(portfolio.tick_data.tvals[2], 5))
    responses += ['-----------------------------------------',
                    '       %s  ||           %s' % (coef1, tval1),
                    '       %s  ||           %s' % (coef2, tval2)]

    for (n_index, number_value) in enumerate(numbers):
        print(f"{number_value:>4}{prompts[n_index]:>25}{responses[n_index]:<3}")
    script_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(script_dir, 'Plots/')
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    portfolio.get_plot_daily_aum(plots_dir)
    portfolio.get_plot_ic(plots_dir)

    return portfolio
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tickers', required=True, help="tickers of the stocks")
    parser.add_argument('--b', type=int, required=True, help="the beginning date of the period")
    parser.add_argument('--e', type=int, help="the ending date of the period")
    parser.add_argument('--initial_aum', type=int,required=True,\
         help="initial asset under management")
    parser.add_argument('--strategy1_type', required=True,\
         help="either 'M'(momentum) or 'R'(reversal)")
    parser.add_argument('--days1', type=int, required=True,\
         help="the numbers of trading days used to compute strategy 1 related returns")
    parser.add_argument('--strategy2_type', required=True,\
         help="either 'M'(momentum) or 'R'(reversal)")
    parser.add_argument('--days2', type=int, required=True,\
         help="the numbers of trading days used to compute strategy 2 related returns")
    parser.add_argument('--top_pct', type=int,required=True,\
         help="an integer from 1 to 100, the percentage of stocks to pick to go long")
    args=parser.parse_args()
    main(args.tickers, args.b, args.e, args.initial_aum,\
         args.strategy1_type, args.days1, args.strategy2_type, args.days2, args.top_pct)
