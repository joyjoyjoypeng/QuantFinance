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
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

class Stock:
    """
    A class used to represent the stock
    ...

    Attributes
    ----------
    tick_data : dictionary
        a formatted dictionary containing key information about the stock portfolio
        per trading day
    beginning_date : date
        the first trading day
    ending_date : date
        the final trading day
    initial_aum : int
        the initial AUM that was invested into the stock
    ticker : string
        the 4 character name of the ticker

    Methods 
    -------
    compute_mm(ticker, start_d, end_d, aum)
        generates the large dictionary of key information
    get_beginning_date()
        gets the first trading date
    get_end_date()
        gets the last trading date
    calc_days()
        calculates the number of calendar days in the trading period
    get_final_value()
        gets the final value of the AUM
    calc_tsr()
        calculates the total stock return taking dividends into account
    calc_tr()
        calculates the total return of the portfolio
    calc_aror(tsr)
        calculates the annualized rate of return of the stock
    calc_avg_aum()
        calculates the average value of the AUM
    get_max_aum()
        gets the maximum value of the AUM
    get_pnl()
        gets the profit and loss of the stock given the AUM
    get_adr()
        gets the average daily return of the stock
    get_std()
        gets the standard deviation of the daily stock return
    get_sharpe()
        gets the Sharpe Ratio of the stock
    get_plot()
        generates the plot of the value of the AUM across the trading period
    """
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
        actual_data (dictionary): The large dictionary. Contains all the relevant data for each
        of the dates within the period in the following order: [closing price, updated AUM value,
        dividends, daily trading rate].
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
        This function gets the first date from the large dictionary.

        Inputs:
        --None, except for the module itself--

        Returns:
        date (string): The first trading date.
        '''
        return list(self.tick_data.keys())[0]

    def get_end_date(self):
        '''
        This function gets the last date from the large dictionary.

        Inputs:
        --None, except for the module itself--

        Returns:
        date (string): The last trading date.
        '''
        return list(self.tick_data.keys())[-1]

    def calc_days(self):
        '''
        This function computes the number of calendar days that have passed during the period
        specified.

        Inputs:
        --None, except for the module itself--

        Returns:
        days (int): Number of calendar days between the first and last trading days.
        '''
        num_days = date.fromisoformat(self.ending_date) - date.fromisoformat(self.beginning_date)
        return num_days.days


    def get_final_value(self):
        '''
        This function returns the final value of the AUM.

        Inputs:
        --None, except for the module itself--

        Returns:
        AUM (float): Final value of the AUM, calculated using the number of initial shares bought
        multiplied by the closing price on the final trading day.
        '''
        return self.tick_data[self.ending_date][1]

    def calc_tsr(self):
        '''
        This function calculates the total stock return using data from the large dictionary.
        This value tells the user how lucrative the stock has been taking into account the
        dividents recieved throughout the specified period.

        Inputs:
        --None, except for the module itself--

        Returns:
        total_stock_return (float): Calculated by taking the final closing price of the stock
        summed with the dividends received throughout the period, and dividing the obtained value
        by the intial stock closing price, and finally subtracting the result by 1.
        '''
        initial_price = self.tick_data[self.beginning_date][0]
        final_price = self.tick_data[self.ending_date][0]
        divs = sum([x[2] for x in list(self.tick_data.values())])
        total_stock_return = (final_price+divs)/initial_price - 1
        return total_stock_return

    def calc_tr(self):
        '''
        This function calculates the total return using data from the large dictionary.
        This value tells the user how lucrative their portfolio has been taking into account
        the starting and final values of their AUM. If more stocks were included in the users
        portfolio or if fractional stocks were not possible, this value would be different
        from the Profit and Loss value of the stock that is also generated later on.

        Inputs:
        --None, except for the module itself--

        Returns:
        total_return (float): Calculated by taking the final value of the AUM and subtracting
        the initial value of the AUM, and finally dividing the obtained value by the initial
        value of the AUM again.
        '''
        initial_aum = self.initial_aum
        end_aum = self.tick_data[self.ending_date][1]
        total_return = (end_aum - initial_aum)/initial_aum
        return total_return

    def calc_aror(self,tsr):
        '''
        This function calcluates the annualized rate of return of the AUM, assuming 250 trading
        days in a year. This value is useful in informing the user on their projected returns if
        the stock were to continue performing in a similar trend to how it has been in the
        specificed period.

        Inputs:
        tsr (float): The total stock return of the stock as calculated earlier.

        Returns:
        annualized rate of return (float): Calculated by taking the time that has passed as a
        percentage of a hypothetical trading year and raising the result as the power of the 
        total stock return subtracted by 1
        '''
        years = len(self.tick_data) / 250
        aror = (tsr + 1)**(1/years) - 1
        return aror

    def calc_avg_aum(self):
        '''
        This function calculates the average value of the AUM throughout the period.

        Inputs:
        --None, except for the module itself--

        Returns:
        average AUM (float): The mean value of the AUM.
        '''
        return np.mean([x[1] for x in list(self.tick_data.values())])

    def get_max_aum(self):
        '''
        This function calculates the maximum value of the AUM throughout the period.

        Inputs:
        --None, except for the module itself--

        Returns:
        max AUM (float): The maximum value of the AUM.
        '''
        return max([x[1] for x in list(self.tick_data.values())])

    def get_pnl(self):
        '''
        This function lets the user know how much profit or loss has been made throughout
        their investment period. The formula used is identical to the total_return function
        earlier since the portfolio only consists of 1 stock with fractional shares.

        Inputs:
        --None, except for the module itself--

        Returns:
        profit_loss (float): Calculated by taking the final value of the AUM and subtracting
        the initial value of the AUM, and finally dividing the obtained value by the initial
        value of the AUM again.
        '''
        initial_aum = self.initial_aum
        end_aum = self.tick_data[self.ending_date][1]
        profit_loss = (end_aum - initial_aum)/initial_aum
        return profit_loss

    def get_adr(self, tsr):
        '''
        This function calculates the average daily return of the stock.

        Inputs:
        tsr (float): The total stock return of the stock as calculated earlier.

        Returns:
        adr (float): Calculated by taking the total stock return and dividing the value by the
        total number of trading days.
        '''
        return tsr / len(self.tick_data)

    def get_std(self):
        '''
        This function calculates the standard deviation of the return of the stock across the
        trading period.

        Inputs:
        --None, except for the module itself--

        Returns:
        std (float): The large dictionary contains the stock return per day, which is now used
        to calculate the standard deviation.
        '''
        return np.std([x[3] for x in list(self.tick_data.values())])

    def get_sharpe(self, adr, std, risk_free = 0.0001):
        '''
        This function calculates the Sharpe Ratio of the stock across the trading period. This
        is one of the most useful statistics as it informs the user of the volatility of the
        stock, acting as a good signal for any further action that needs to be taken.

        Inputs:
        adr (float): The average daily return of the stock as calculated earlier.
        std (float): The standard deviation of the daily stock return as calculated earlier.
        risk_free (float): A value of 0.01% is assumed and can be modified if necessary.

        Returns:
        Sharpe Ratio (float): Calculated by taking the average daily return, subtracting the risk
        free rate and dividing the result by the standard deviation of the daily stock return
        '''
        return (adr-risk_free)/std

    def get_plot(self):
        '''
        This function generates the plot of the value of the AUM across the trading period if
        requested by the user.

        Inputs:
        --None, except for the module itself--

        Returns:
        the plot (plot): The plot is shown to the user directly, and is also saved to their
        local drive. If the user wishes to run the script again using a different ticker, the new
        plot will not override the previous one as it will be saved under a different name.
        '''

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
        plt.show()


def main (ticker, b_date, e_date, initial_aum, plot=None):
    '''
    This is the main function for initializing, computing, and retrieving the final output.
    The printed outputs are also generated here, color coded according to their negative/positive
    values.

    Inputs:
    ticker (string): 4 character ticker name, as provided by the user
    b_date (string): The date the initial AUM was invested
    e_date (string): The final date to be used for calculations. If no ending date is
    provided by the user, the script takes the latest possible date.
    initial_aum (integer): The intial amount of assets that are invested in the stock
    plot (bool): Whether a plot should be generated showing the value of the AUM across the
    trading period

    Returns:
    stock (class): The module containing all the relevant information about the stock.
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
    print("Still generating...")
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
    if np.equal(plot, True):
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
