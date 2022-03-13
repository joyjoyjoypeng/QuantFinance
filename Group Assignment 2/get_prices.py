'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 13 Mar 2022

This program fetches daily close prices from the internet for a given ticker and time frame,
and computes some analytics on them.

The program can be called using:
get_prices.py --ticker <xxx> –-b <YYYYMMDD> –-e <YYYYMMMDD> --initial_aum <yyyy>  --plot
'''

import argparse
from datetime import date
import numpy as np
import yfinance as yf
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

class Stock:
    '''
        This is the class function for Stock objects

        Inputs:
        (wait joy idk whats supposed to go here i m sorry pls help)

        Returns:
        Everything.
    '''
    def __init__(self, ticker, beginning_date, ending_date, aum):
        '''
        This function initializes the inputs given by the user and stores them within the class
        so that they can be used by the other functions. It also creates the "tick_data" which
        contains the key information about the ticker given the start and end dates.

        Inputs:
        ticker (string): 4 character ticker name, as provided by the user
        beginning_date (string): The date the initial AUM was invested
        ending_date (string): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.
        aum (integer): The intial amount of assets that are invested in the stock

        Returns:
        --Nothing is returned as the data will be stored within the class--
        '''
        self.tick_data=self.compute_mm(ticker, beginning_date, ending_date, aum)
        self.beginning_date=self.get_beginning_date()
        self.ending_date=self.get_end_date()
        self.initial_aum=aum
        self.ticker_name=ticker

    def compute_mm(self, ticker, start_d, end_d, aum):
        '''
        This function pulls all available closing price and dividends data for the ticker
        provided before zooming in to the relevant period using the start and ending dates.
        It also tracks the value of the AUM given the inital number of shares bought and the
        daily trading rate throughout the period for easier computations in future functions.

        Inputs:
        ticker (string): 4 character ticker name, as provided by the user
        start_d (date): The date the initial AUM was invested
        end_d (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.
        aum (integer): The intial amount of assets that are invested in the stock

        Returns:
        actual_data (dictionary): Contains all the relevant data for each of the dates within
        the period in the following order: [closing price, updated AUM value, dividends, daily
        trading rate].
        '''
        ticker_data = yf.Ticker(ticker).history(period='max')
        actual_data = {}
        for (close_index, close_value) in enumerate(ticker_data['Close']):
            cur_d = date.fromisoformat(str(ticker_data['Close'].index[close_index]).split(' ')[0])
            if (cur_d-start_d).days>=0 and (cur_d-end_d).days<=0:
                if len(actual_data) == 0:
                    shares=aum / close_value
                    actual_data[str(cur_d)]=[close_value, aum, 
                    ticker_data['Dividends'][close_index], 0]
                else:
                    daily_trading_rate = (close_value + ticker_data['Dividends'][close_index]) /\
                    ticker_data['Close'][close_index-1] - 1
                    actual_data[str(cur_d)]=[close_value, shares * close_value, 
                    ticker_data['Dividends'][close_index], daily_trading_rate]
        return actual_data

    def get_beginning_date(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        return list(self.tick_data.keys())[0]

    def get_end_date(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        return list(self.tick_data.keys())[-1]

    def calc_days(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        num_days = date.fromisoformat(self.ending_date) - date.fromisoformat(self.beginning_date)
        return num_days.days


    def get_final_value(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        return self.tick_data[self.ending_date][1]

    def calc_tsr(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        initial_price = self.tick_data[self.beginning_date][0]
        final_price = self.tick_data[self.ending_date][0]
        divs = sum([x[2] for x in list(self.tick_data.values())])
        total_stock_return = (final_price+divs)/initial_price - 1
        return total_stock_return

    def calc_tr(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        initial_aum = self.initial_aum
        end_aum = self.tick_data[self.ending_date][1]
        total_return = (end_aum - initial_aum)/initial_aum
        return total_return

    def calc_aror(self,tsr):
        '''
        This function

        Inputs:


        Returns:
        '''
        years = len(self.tick_data) / 250
        aror = (tsr + 1)**(1/years) - 1
        return aror * self.initial_aum

    def calc_avg_aum(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        return np.mean([x[1] for x in list(self.tick_data.values())])

    def get_max_aum(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        return max([x[1] for x in list(self.tick_data.values())])

    def get_pnl(self):
        '''
        This function

        Inputs:


        Returns:
        '''
        initial_aum = self.initial_aum
        end_aum = self.tick_data[self.ending_date][1]
        profit_loss = (end_aum - initial_aum)/initial_aum
        return profit_loss

    def get_adr(self, tsr):
        '''
        This function

        Inputs:


        Returns:
        '''
        return tsr / len(self.tick_data)

    def get_std(self):
        return np.std([x[3] for x in list(self.tick_data.values())])

    def get_sharpe(self, adr, std, risk_free = 0.0001):
        return (adr-risk_free)/std

    def get_plot(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 15))
        plt.xlabel("Date")
        plt.ylabel("AUM")
        plt.title("AUM Across Time (" + self.ticker_name + ")")

        data_frame = pd.DataFrame.from_dict(self.tick_data, orient = 'index')
        data_frame = data_frame.reset_index()
        data_frame["index"] = data_frame["index"].astype("datetime64")
        data_frame = data_frame.set_index("index")

        plt.plot(data_frame[1])
        plt.xticks(rotation=45)
        plt.savefig("Plot of AUM for " + self.ticker_name + ".jpg")


def main (ticker, b_date, e_date, initial_aum, plot=None):
    '''
    This is the main function for initializing, computing, and retrieving the final output.

    Inputs:

    Returns:
    '''
    if yf.Ticker(ticker).info['regularMarketPrice'] is None:
        raise NameError("No stock ticker name found.")
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
                raise ValueError('The ending date is after today\'s date. No data available.')
    except ValueError as v_e:
        raise ValueError('You have entered an invalid date.') from v_e 
    stock = Stock(ticker, beginning_date, ending_date, initial_aum)
    days = stock.calc_days()
    tsr = stock.calc_tsr()
    total_return = stock.calc_tr()
    aror= stock.calc_aror(tsr)
    final_aum = stock.get_final_value()
    average_aum = stock.calc_avg_aum()
    max_aum = stock.get_max_aum()
    pnl = stock.get_pnl()
    adr = stock.get_adr(tsr)
    std = stock.get_std()
    sharpe = stock.get_sharpe(adr, std)
    if plot == True:
        stock.get_plot()
    numbers = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "(10)", "(11)",
    "(12)", "(13)", "(14)"]
    prompts = ["Begin date:  ", "Ending date:  ", "Number of days:  ", "Total Stock Return:  ",
    "Total Return:  ", "Annualized RoR:  ", "Initial AUM:  ", "Final AUM:  ", "Average AUM:  ",
    "Maximum AUM:  ", "PnL:  ", "Average Daily Return:  ", "Daily SD of Return:  ",
    "Daily Sharpe Ratio:  "]
    responses = [stock.beginning_date, stock.ending_date, days, tsr, total_return, aror,
    initial_aum, final_aum, average_aum, max_aum, pnl, adr, std, sharpe]
    colors = []
    for (n_index, n_val) in enumerate(numbers):
        if n_val in ['(4)', '(5)', '(6)', '(11)', '(12)', '(14)']:
            if responses[n_index] < 0:
                colors.append('red')
            else:
                colors.append('green')
        else:
            colors.append('white')
    for (n_index, number_value) in enumerate(numbers):
        print(f"{number_value:>4}{prompts[n_index]:>25}",\
        colored(f"{responses[n_index]:<3}", f"{colors[n_index]}"))
    return stock

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True, help="ticker of the stock")
    parser.add_argument('--b',type=int, required=True, help="the beginning date of the period")
    parser.add_argument('--e', type=int, help="the ending date of the period")
    parser.add_argument('--initial_aum', type=int, required=True,help="number of k-folds")
    parser.add_argument('--plot', action='store_true', help="a plot of the the daily AUM")
    args=parser.parse_args()
    main(args.ticker, args.b, args.e, args.initial_aum, args.plot)
