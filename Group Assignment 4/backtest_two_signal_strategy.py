'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 4 May 2022

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
import warnings
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
warnings.filterwarnings('ignore')

class Ticker:
    """
    A class used to contain information about all the stocks that are part of the user's
    portfolio, given their specificed dates and strategy information

    Attributes
    ----------
    giant : dictionary
        A giant dictionary that contains all the information about each stock for each trading
        day within the specified period, such as its closing price, dividends
    dates : list
        The dates on which trades are made, i.e. the last trading days of each month within
        the specified period
    predates : list
        Same as dates, but also contains the last day of the trading month before the start
        date so that the linear model can be fitted for the first day of buying stocks
    strnum :
        The number of stocks that need to be picked from the portfolio each time, obtained
        using the top_pct value provided by the user
    strat : int
        The strategy that will be employed in the backtesting, specifying the stocks that
        should be bought on each trading day
    tvals : list
        The t-values of the coefficients of each stock's final linear model
    coefs : list
        The coefficients of each stock's final linear model for each strategy's returns
    trading_data : dictionary
        A slightly smaller dictionary that only contains data for the days on which the
        user has ownership of the stock based on the trading strategy, and contains data
        on the number of shares that are owned and the value of the AUM invested in that stock

    Methods
    -------
    compute_giant_date(tickers_list, beginning_date, ending_date):
        This function pulls data for all of the stocks provided and fetches the dates of
        the last trading days of each month
    compute_strategy(strategy1_type, days1, strategy2_type, days2, tickers,list):
        This function performs linear regression to build on past data and generate coefficients
        of the different strategy types to best predict the stocks to go long with for the
        next month
    compute_trading_data(tickers_list, beginning_date, ending_date, aum):
        This function generates the slightly smaller dictionary of key information, such as the
        number of shares owned, the dividends the user will received, value of AUM in each stock
    compute_total(aum):
        This function generates an even smaller dictionary containing the AUM value of the user
        across all dates, and the day-on-day return for the user's portfolio
    """
    def __init__(self, tickers_list, beginning_date, ending_date, aum, strategy1_type,\
        days1, strategy2_type, days2, top_pct):
        self.giant=self.compute_giant_date(tickers_list, beginning_date, ending_date)[0]
        self.predates=self.compute_giant_date(tickers_list, beginning_date, ending_date)[1]
        self.dates=self.predates[1:]
        self.strnum = ceil(len(tickers_list) * (top_pct/100))
        print('Computing strategy...')
        (self.strat, self.tvals, self.coefs)=self.compute_strategy(\
            strategy1_type, days1, strategy2_type, days2, tickers_list)
        print('Generating returns...')
        self.trading_data=self.compute_trading_data(tickers_list,beginning_date,ending_date,aum)[0]
    def compute_giant_date(self, tickers_list, beginning_date, ending_date):
        '''
        This function pulls all available closing price and dividends data for the tickers
        provided before zooming in to the relevant period using the start and ending dates.
        It also fetches the dates of the last trading days for each month

        Inputs:
        tickers_list (list): a list of the comma-separated 4-character ticker names, as provided
        by the user
        beginning_date (date): The date the initial AUM was invested
        ending_date (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date

        Returns:
        giant (dictionary): contains all the information of each stock in the specified period
        dates (list): contains all the trading days + the last trading day of the month before
        the start date
        '''
        giant = {}
        dates = []
        comp = 'start'

        for ticker in tickers_list:
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(start = beginning_date-timedelta(450), end = ending_date)
            giant[ticker] = data
            if len(dates) == 0:
                for row in range(len(data['Close'])):
                    cur_date = data['Close'].index[row].date()
                    if comp == 'start':
                        comp = cur_date
                    elif comp.month != cur_date.month:
                        if comp >= beginning_date and comp <= ending_date:
                            dates.append(comp)
                        elif comp.month + 1 == beginning_date.month:
                            dates.append(comp)
                        elif comp.year + 1 == beginning_date.year and \
                            comp.month - 11 == beginning_date.month:
                            dates.append(comp)
                    comp = cur_date
        return (giant, dates)
    def compute_strategy(self, strategy1_type, days1, strategy2_type, days2, tickers_list):
        '''
        This function performs linear regression to calculate which stocks should be bought
        on each trading datafter weighing the returns from utilizing each strategy provided
        by the user

        Inputs:
        strategy1_type (str): the first backtesting strategy chosen by the user
        strategy2_type (str): same as above, but the second one
        days1 (int): the first numbers of trading days used to compute strategy-related returns
        days2 (int): same as above, but the second one
        tickers_list (list): a list of the 4-character ticker names, as provided by the user

        Returns:
        strat_ticks (dictionary): contains the list of stocks that need to be bought
        on each trading day
        '''
        strat = {}
        final_prices = {}
        linear_xvals = []
        linear_yvals = []

        for trading_day in self.predates:
            for tick, data in self.giant.items():
                if tick not in final_prices:
                    final_prices[tick] = []
                for (row_ind, row_val) in enumerate(data['Close']):
                    cur_date = data['Close'].index[row_ind].date()
                    if cur_date == trading_day:
                        final_price = row_val
                        if strategy1_type == 'R':
                            endp_1 = row_val
                            sttp_1 = data['Close'][row_ind-days1]
                            divp_1 = 0
                            for day in range(row_ind-days1, row_ind+1):
                                divp_1 += data['Dividends'][day]
                        else:
                            endp_1 = data['Close'][row_ind-20]
                            sttp_1 = data['Close'][row_ind-20-days1]
                            divp_1 = 0
                            for day in range(row_ind-20-days1, row_ind-19):
                                divp_1 += data['Dividends'][day]
                        ret_1 = (endp_1 - sttp_1 + divp_1) / sttp_1
                        if strategy2_type == 'R':
                            endp_2 = row_val
                            sttp_2 = data['Close'][row_ind-days2]
                            divp_2 = 0
                            for day in range(row_ind-days2, row_ind+1):
                                divp_2 += data['Dividends'][day]
                        else:
                            endp_2 = data['Close'][row_ind-20]
                            sttp_2 = data['Close'][row_ind-20-days2]
                            divp_2 = 0
                            for day in range(row_ind-20-days2, row_ind-19):
                                divp_2 += data['Dividends'][day]
                        ret_2 = (endp_2 - sttp_2 + divp_2) / sttp_2
                        linear_xvals.append([ret_1, ret_2])
                        final_prices[tick].append([final_price, row_ind])
                        if len(linear_xvals) > len(tickers_list):
                            if trading_day not in strat:
                                strat[trading_day] = {}
                            endp = row_val
                            sttp = final_prices[tick][-2][0]
                            divp = 0
                            for day in range(final_prices[tick][-2][1], row_ind+1):
                                divp += data['Dividends'][day]
                            ret = (endp - sttp + divp) / sttp
                            linear_yvals.append(ret)
                            if len(linear_yvals) % len(tickers_list) == 0 and \
                                len(linear_yvals) != 0:
                                linear_model = LinearRegression().fit(\
                                    linear_xvals[:-len(tickers_list)], linear_yvals)
                                for ind, tick in enumerate(tickers_list):
                                    values = linear_xvals[ind-len(tickers_list)]
                                    pred = linear_model.predict([values])[0]
                                    strat[trading_day][tick] = pred

        linear_coefs= linear_model.coef_
        xvals = sm.add_constant(linear_xvals[:-len(tickers_list)])
        new_model = sm.OLS(linear_yvals, xvals)
        tval_model = new_model.fit()
        linear_tvals = tval_model.summary2().tables[1]['t']

        strat_ticks = {}

        for trading_day, stats in strat.items():
            strat_ticks[trading_day] = []
            ranked = dict(sorted(stats.items(), key=lambda item: item[1]))
            num = 0
            while num != self.strnum:
                strat_ticks[trading_day].append(list(ranked.keys())[num])
                num += 1

        return (strat_ticks, linear_tvals, linear_coefs)

    def compute_trading_data(self, tickers_list, beginning_date, ending_date, aum):
        '''
        This function uses the strategy generated to buy and sell stocks accordingly,
        tracking the number of shares owned for each stock and the value of the AUM.
        It also keeps track of the information coefficient

        Inputs:
        tickers_list (list): a list of the 4-character ticker names, as provided by the user
        beginning_date (date): the date the initial AUM was invested
        ending_date (date): the final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date
        aum (integer): the intial amount of assets that are invested in the portfolio

        Returns:
        trading_data (dictionary): the large dictionary. Contains all the relevant data for each
        of the dates within the period
        ic_values (dictionary): a dictionary tracking the changes in the information coefficient
        throughout the trading days, based on the perfomance of the stocks predicted by the
        strategy
        '''
        trading_data = {}
        ic_values = {}
        for row in range(len(self.giant[tickers_list[0]]['Close'])):
            cur_date = self.giant[tickers_list[0]]['Close'].index[row].date()
            if cur_date >= beginning_date and cur_date <= ending_date:
                if cur_date in self.dates:
                    if len(trading_data) == 0:
                        trading_data[str(cur_date)] = {}
                        for tick in tickers_list:
                            if tick in self.strat[cur_date]:
                                shares = (aum/self.strnum) / self.giant[tick]['Close'][row]
                                trading_data[str(cur_date)][tick] = \
                                [self.giant[tick]['Close'][row],(aum/self.strnum), shares,
                                self.giant[tick]['Dividends'][row]]
                            else:
                                trading_data[str(cur_date)][tick] = \
                                [self.giant[tick]['Close'][row],0, 0, 0]
                    else:
                        trading_data[str(cur_date)] = {}
                        prev = self.giant[tickers_list[0]]['Close'].index[row-1].date()
                        new_aum = 0
                        for tick in tickers_list:
                            new_aum+=\
                            trading_data[str(prev)][tick][2]*self.giant[tick]['Close'][row]
                        if new_aum == 0:
                            new_aum += aum

                        for tick in tickers_list:
                            if tick in self.strat[cur_date]:
                                shares = (new_aum/self.strnum) / self.giant[tick]['Close'][row]
                                trading_data[str(cur_date)][tick] = \
                                [self.giant[tick]['Close'][row],(new_aum/self.strnum), shares,
                                self.giant[tick]['Dividends'][row]]
                            else:
                                trading_data[str(cur_date)][tick] = \
                                [self.giant[tick]['Close'][row], 0, 0, 0]

                        pos = self.dates.index(cur_date)
                        if pos != 0:
                            ic_values[cur_date] = 0
                            prem = self.dates[pos-1]
                            for tick in tickers_list:
                                if tick in self.strat[prem]:
                                    if trading_data[str(cur_date)][tick][0]>\
                                    trading_data[str(prem)][tick][0]:
                                        ic_values[cur_date] += 1
                            ic_values[cur_date] = [2 * (ic_values[cur_date]/self.strnum) - 1]
                else:
                    if len(trading_data) == 0:
                        trading_data[str(cur_date)] = {}
                        for tick in tickers_list:
                            trading_data[str(cur_date)][tick] = [self.giant[tick]['Close'][row],
                            0, 0, 0]
                    else:
                        prev = self.giant[tickers_list[0]]['Close'].index[row-1].date()
                        trading_data[str(cur_date)] = {}
                        for tick in tickers_list:
                            shares = trading_data[str(prev)][tick][2]
                            trading_data[str(cur_date)][tick] = [self.giant[tick]['Close'][row],
                            shares * self.giant[tick]['Close'][row], shares,
                            self.giant[tick]['Dividends'][row]]
        tot = 0
        for trading_day, val in ic_values.items():
            tot += val[0]
            ic_values[trading_day].append(tot)
        return (trading_data,ic_values)
    def compute_total(self, aum):
        '''
        This function keeps track of the users combined AUM across the amounts invested
        in each stock and the daily return

        Inputs:
        aum (integer): The intial amount of assets that are invested in the portfolio

        Returns:
        total (dictionary): contains the updated value of the AUM on each day and the return
        '''
        total = {}
        place = 0
        for cur_day, stocks in self.trading_data.items():
            newaum = 0
            for value in stocks.values():
                newaum += value[1]
            if place != 0:
                dpr = (newaum-place)/place
            else:
                dpr = 0
            if newaum == 0:
                newaum = aum
                dpr = 0
            total[cur_day] = [newaum, dpr]
            place = newaum
        return total
class Portfolio:
    """
    A class used to represent the user's portfolio
    ...

    Attributes
    ----------
    tick_data : class
        all the attributes of the Ticker class
    total : dictionary
        from the Ticker class, a dictionary tracking the user's daily AUM and return
    beginning_date : date
        the first trading day
    ending_date : date
        the final trading day
    ic_values : dictionary
        from the Ticker class, a dictionary tracking the IC across trading days

    Methods
    -------

    get_beginning_date()
        gets the first date
    get_end_date()
        gets the last date
    calc_num_of_days(start, end):
        calculates the number of calendar days in the period specified
    get_final_value(end):
        gets the final value of the AUM
    def calc_TSR(start, end, tick):
        calculates the total stock return of each stock taking dividends into account
    calc_TR(start, end):
        calculates the total return of the portfolio
    calc_aror(tr)
        calculates the annualized rate of return of the portfolio
    calc_avg_aum()
        calculates the average value of the total AUM
    get_max_aum()
        gets the maximum value of the total AUM
    get_pnl(start, end)
        gets the profit and loss of the stock given the total AUM
    get_adr(tr)
        gets the average daily return of the portfolio
    get_std()
        gets the standard deviation of the daily portfolio return
    get_sharpe(adr, std, risk_free = 0.0001)
        gets the Sharpe Ratio of the portfolio
    get_plot_daily_aum():
        Generates a plot that shows the daily AUM thru time
    get_plot_ic():
        Generates a plot that shows the monthly cumulative information coefficient
    """
    def __init__(self,tickers_list, beginning_date, ending_date, aum, strategy1_type,\
        days1, strategy2_type, days2, top_pct):
        self.tick_data=Ticker(tickers_list,beginning_date,ending_date,
        aum,strategy1_type,days1,strategy2_type,days2,top_pct)
        self.total=self.tick_data.compute_total(aum)
        self.beginning_date=self.get_start_date()
        self.ending_date=self.get_end_date()
        self.ic_values=self.tick_data.compute_trading_data(tickers_list,beginning_date,\
        ending_date,aum)[1]
    def get_start_date(self):
        '''
        This function gets the first date from the large dictionary.

        Inputs:
        --None, except for the module itself--

        Returns:
        date (string): the first trading date
        '''
        return list(self.total.keys())[0]

    def get_end_date(self):
        '''
        This function gets the last date from the large dictionary.

        Inputs:
        --None, except for the module itself--

        Returns:
        date (string): the last trading date
        '''
        return list(self.total.keys())[-1]

    def calc_num_of_days(self, start, end):
        '''
        This function computes the number of calendar days that have passed during the period
        specified.

        Inputs:
        start (date): the date the initial AUM was invested
        end (date): the final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date

        Returns:
        days (int): number of calendar days between the first and last trading days
        '''
        days = date.fromisoformat(end) - date.fromisoformat(start)
        return days.days

    def get_final_value(self, end):
        '''
        This function returns the final value of the AUM.

        Inputs:
        end (date): the final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date

        Returns:
        AUM (float): final value of the AUM, obtained from the "total" dictionary
        '''
        return self.total[end][0]

    def calc_tsr(self, tick):
        '''
        This function calculates the total stock return using data from the large dictionary.
        This value tells the user how lucrative the stock has been taking into account the
        dividends recieved throughout the specified period, only when the user was in possession
        of the stock as per the trading strategy. If the stock was not suggested by the trading
        strategy throughout the trading period, it shall have no return

        Inputs:
        tick (string): 4 character ticker name

        Returns:
        total_stock_return (float): calculated by taking the last closing price of the stock
        when the user was in possession of it, summed with the dividends received throughout
        the period (if any), and dividing the obtained value by the intial stock closing price
        when the user was in possession of it, and finally subtracting the result by 1
        '''
        initial_price = 0
        final_price = 0

        for vals in self.tick_data.trading_data.values():
            for cur_tick, values in vals.items():
                if cur_tick == tick:
                    if values[2] > 0:
                        if initial_price == 0:
                            initial_price = values[0]
                        else:
                            final_price = values[0]

        divs = sum([x[tick][2] for x in list(self.tick_data.trading_data.values())])
        tsr = 0
        if initial_price != 0:
            tsr = (final_price+divs)/initial_price - 1
        return tsr

    def calc_tr(self, start, end):
        '''
        This function calculates the total return using data from the large dictionary.
        This value tells the user how lucrative their portfolio has been taking into account
        the starting and final values of their AUM. If fractional stocks were not possible,
        this value would be different from the Profit and Loss value of the portfolio that
        is also generated later on.

        Inputs:
        start (date): the date the initial AUM was invested
        end (date): the final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date

        Returns:
        total_return (float): calculated by taking the final value of the AUM and subtracting
        the initial value of the AUM, and finally dividing the obtained value by the initial
        value of the AUM again
        '''
        initial_aum = self.total[start][0]
        end_aum = self.total[end][0]
        total_return = (end_aum - initial_aum)/initial_aum
        return total_return

    def calc_aror(self, total_return):
        '''
        This function calcluates the annualized rate of return of the AUM, assuming 250 trading
        days in a year. This value is useful in informing the user on their projected returns if
        the stock were to continue performing in a similar trend to how it has been in the
        specificed period.

        Inputs:
        total_return (float): the total return of the portfolio as calculated earlier

        Returns:
        annualized rate of return (float): calculated by taking the time that has passed as a
        percentage of a hypothetical trading year and raising the result as the power of the
        total portfolio return subtracted by 1
        '''
        years = len(self.total) / 250
        aror = (total_return + 1)**(1/years) - 1
        return aror

    def calc_avg_aum(self):
        '''
        This function calculates the average value of the AUM throughout the period

        Inputs:
        --None, except for the module itself--

        Returns:
        average AUM (float): the mean value of the AUM
        '''
        return np.mean([x[0] for x in list(self.total.values())])

    def get_max_aum(self):
        '''
        This function calculates the maximum value of the AUM throughout the period.

        Inputs:
        --None, except for the module itself--

        Returns:
        max AUM (float): the maximum value of the AUM
        '''
        return max([x[0] for x in list(self.total.values())])

    def get_pnl(self, start, end):
        '''
        This function lets the user know how much profit or loss has been made throughout
        their investment period. The formula used is identical to the total_return function
        earlier since the portfolio consists of fractional shares.

        Inputs:
        start (date): the date the initial AUM was invested
        end (date): the final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date

        Returns:
        profit_loss (float): calculated by taking the final value of the AUM and subtracting
        the initial value of the AUM, and finally dividing the obtained value by the initial
        value of the AUM again
        '''
        initial_aum = self.total[start][0]
        end_aum = self.total[end][0]
        pnl = (end_aum - initial_aum)/initial_aum
        return pnl

    def get_adr(self, total_return):
        '''
        This function calculates the average daily return of the portfoilo

        Inputs:
        total_return (float): The total return of the portfolio as calculated earlier

        Returns:
        adr (float): calculated by taking the total portfolio return and dividing the value by the
        total number of days in the period specificed
        '''
        adr = total_return / len(self.total)
        return adr

    def get_std(self):
        '''
        This function calculates the standard deviation of the return of the portfolio across the
        period specified

        Inputs:
        --None, except for the module itself--

        Returns:
        std (float): the "total" dictionary contains the portfolio return per day, which is now
        used to calculate the standard deviation
        '''
        return np.std([x[1] for x in list(self.total.values()) if x[1] != 0])

    def get_sharpe(self, adr, std, risk_free = 0.0001):
        '''
        This function calculates the Sharpe Ratio of the portfolio across the period specified.
        This is one of the most useful statistics as it informs the user of the volatility of the
        portfolio, acting as a good signal for any further action that needs to be taken.

        Inputs:
        adr (float): the average daily return of the portfolio as calculated earlier.
        std (float): the standard deviation of the daily portfolio return as calculated earlier.
        risk_free (float): a value of 0.01% is assumed and can be modified if necessary.

        Returns:
        Sharpe Ratio (float): calculated by taking the average daily return, subtracting the risk
        free rate and dividing the result by the standard deviation of the daily portfolio return
        '''
        sharpe_ratio = (adr-risk_free)/std
        return sharpe_ratio
    def get_plot_daily_aum(self, plots_dir):
        '''
        This function generates the plot of the value of the AUM across the trading period.

        Inputs:
        plots_dir (path): the path for the Plots folder where the plot will be saved

        Returns:
        the plot (jpg): the plot is saved to the user's local drive. If the user wishes to run
        the script again using a different set of tickers, the new plot will override the previous
        one under the same name
        '''
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10, 10))
        plt.xlabel("Date")
        plt.ylabel("AUM")
        plt.title("Daily AUM Across Time")
        aum_vals = pd.DataFrame.from_dict(self.total, orient = 'index')
        aum_vals = aum_vals.reset_index()
        aum_vals["index"] = aum_vals["index"].astype("datetime64")
        aum_vals = aum_vals.set_index("index")
        plt.plot(aum_vals[0])
        plt.xticks(rotation=45)
        plt.savefig(plots_dir + "Daily AUM.jpg")
        plt.show()
    def get_plot_ic(self, plots_dir):
        '''
        This function generates the plot of the monthly cummulative IC (Information Coefficient)

        Inputs:
        plots_dir (path): the path for the Plots folder where the plot will be saved

        Returns:
        the plot (jpg): the plot is saved to the user's local drive. If the user wishes to run
        the script again using a different set of tickers, the new plot will override the previous
        one under the same name
        '''
        plt.figure(figsize=(10, 10))
        plt.xlabel("Date")
        plt.ylabel("Cumm IC")
        plt.title("Cumm IC Across Time")
        info_vals = pd.DataFrame.from_dict(self.ic_values, orient = 'index')
        info_vals = info_vals.reset_index()
        info_vals["index"] = info_vals["index"].astype("datetime64")
        info_vals = info_vals.set_index("index")
        plt.plot(info_vals[1])
        plt.xticks(rotation=45)
        plt.savefig(plots_dir + "Cummulative IC.jpg")
        plt.show()
def main (tickers,b_date,e_date,initial_aum,strategy1_type,days1,strategy2_type,days2,top_pct):
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
