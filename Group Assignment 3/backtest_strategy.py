'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 14 Apr 2022

This program fetches daily close prices from the internet for given tickers and time frame,
and then back tests some simple momentum and reversal monthly strategies.

The program can be called using:
backtest_strategy.py --tickers <ttt> –-b <YYYYMMDD> –-e <YYYYMMDD> --initial_aum <yyyy>
--strategy_type <xx> --days <xx> --top_pct <xx>
'''
import argparse
from datetime import date, timedelta
from math import ceil
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
class Ticker:
    """
    A class used to represent the individual stock
    ...

    Attributes
    ----------
    giant : dictionary
        ...
    dates : dictionary
        ...
    strnum :
        ...
    strat : int
        ...
    trading_data : dictionary
        ...

    Methods
    -------
    compute_giant_date(tickers_list, beginning_date, ending_date):
        ...
    compute_strategy(strategy_type, days):
        ...
    compute_trading_data(tickers_list, beginning_date, ending_date, aum):
        generates the large dictionary of key information
    compute_total(aum):
    """
    def __init__(self,tickers_list,beginning_date,ending_date,aum,strategy_type,days,top_pct):
        self.giant=self.compute_giant_date(tickers_list, beginning_date, ending_date)[0]
        self.dates=self.compute_giant_date(tickers_list, beginning_date, ending_date)[1]
        self.strnum = ceil(len(tickers_list) * (top_pct/100))
        self.strat=self.compute_strategy(strategy_type, days)
        self.trading_data=self.compute_trading_data(tickers_list,beginning_date,ending_date,aum)[0]
    def compute_giant_date(self, tickers_list, beginning_date, ending_date):
        '''
        This function ...

        Inputs:
        tickers_list (list): a list of the 4-character ticker names, as provided by the user
        beginning_date (date): The date the initial AUM was invested
        ending_date (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.

        Returns:
        giant (dictionary): ...
        dates:...
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
                    comp = cur_date
        return (giant, dates)
    def compute_strategy(self, strategy_type, days):
        '''
        This function...

        Inputs:
        strategy_type: The back testing stratefy chosen by the user.
        days: The numbers of trading days used to compute strategy-related returns.

        Returns:
        strat_ticks (dictionary): ...
        '''
        strat = {}

        for trading_day in self.dates:
            strat[trading_day] = {}
            for tick, data in self.giant.items():
                for row in range(len(data['Close'])):
                    cur_date = data['Close'].index[row].date()
                    if cur_date == trading_day:
                        if strategy_type == 'R':
                            endp = data['Close'][row]
                            sttp = data['Close'][row-days]
                        else:
                            endp = data['Close'][row-20]
                            sttp = data['Close'][row-20-days]
                        ret = (endp - sttp) / sttp
                        strat[trading_day][tick] = ret

        strat_ticks = {}

        for trading_day, stats in strat.items():
            strat_ticks[trading_day] = []
            ranked = dict(sorted(stats.items(), key=lambda item: item[1]))
            num = 0
            while num != self.strnum:
                strat_ticks[trading_day].append(list(ranked.keys())[num])
                num += 1

        return strat_ticks
    #add def for ic
    def compute_trading_data(self, tickers_list, beginning_date, ending_date, aum):
        '''
        This function pulls all available closing price and dividends data for the ticker
        provided before zooming in to the relevant period using the start and ending dates.
        It also tracks the value of the AUM given the inital number of shares bought and the
        daily trading rate throughout the period for easier computations in future functions.

        Inputs:
        tickers_list (list): a list of the 4-character ticker names, as provided by the user
        beginning_date (date): The date the initial AUM was invested
        ending_date (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.
        aum (integer): The intial amount of assets that are invested in the stock

        Returns:
        trading_data (dictionary): The large dictionary. Contains all the relevant data for each
        of the dates within the period in the following order: [closing price, updated AUM value,
        dividends, daily trading rate].
        ic_values:...
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
                                trading_data[str(cur_date)][tick] = [self.giant[tick]['Close'][row],
                                (aum/self.strnum), shares, self.giant[tick]['Dividends'][row]]
                            else:
                                trading_data[str(cur_date)][tick] = [self.giant[tick]['Close'][row],
                                0, 0, self.giant[tick]['Dividends'][row]]
                    else:
                        trading_data[str(cur_date)] = {}
                        prev = self.giant[tickers_list[0]]['Close'].index[row-1].date()
                        new_aum = 0
                        for tick in tickers_list:
                            new_aum+=trading_data[str(prev)][tick][2]*self.giant[tick]['Close'][row]
                        if new_aum == 0:
                            new_aum += aum

                        for tick in tickers_list:
                            if tick in self.strat[cur_date]:
                                shares = (new_aum/self.strnum) / self.giant[tick]['Close'][row]
                                trading_data[str(cur_date)][tick] = [self.giant[tick]['Close'][row],
                                (new_aum/self.strnum), shares, self.giant[tick]['Dividends'][row]]
                            else:
                                trading_data[str(cur_date)][tick] = [self.giant[tick]['Close'][row],
                                0, 0, self.giant[tick]['Dividends'][row]]

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
                            0, 0, self.giant[tick]['Dividends'][row]]
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
        This function...

        Inputs:
        aum (integer): The intial amount of assets that are invested in the stock

        Returns:
        total (dictionary): ...
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
    A class used to represent the portfolio of stocks
    ...

    Attributes
    ----------
    tick_data : dictionary
        a formatted dictionary containing key information about the stock portfolio
        per trading day
    total : dictionary
        ...
    beginning_date : date
        the first trading day
    ending_date : date
        the final trading day
    ic_values : dictionary
        ...

    Methods
    -------

    get_beginning_date()
        gets the first trading date
    get_end_date()
        gets the last trading date
    calc_num_of_days(start, end):
        calculates the number of calendar days in the trading period
    get_final_value(end):
        gets the final value of the AUM
    def calc_TSR(start, end, tick):
        calculates the total stock return taking dividends into account
    calc_TR(start, end):
        calculates the total return of the portfolio
    calc_aror(tr)
        calculates the annualized rate of return of the stock
    calc_avg_aum()
        calculates the average value of the AUM
    get_max_aum()
        gets the maximum value of the AUM
    get_pnl(start, end)
        gets the profit and loss of the stock given the AUM
    get_adr(tr)
        gets the average daily return of the stock
    get_std()
        gets the standard deviation of the daily stock return
    get_sharpe(adr, std, risk_free = 0.0001)
        gets the Sharpe Ratio of the stock
    get_plot_daily_aum():
        Generates a plot that shows the daily AUM thru time
    get_plot_ic():
        Generates a plot that shows the monthly cumulative information coefficient
    """
    def __init__(self,tickers_list,beginning_date,ending_date,aum,strategy_type,days,top_pct):
        self.tick_data=Ticker(tickers_list,beginning_date,ending_date,
        aum,strategy_type,days,top_pct)
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
        date (string): The first trading date.
        '''
        return list(self.total.keys())[0]

    def get_end_date(self):
        '''
        This function gets the last date from the large dictionary.

        Inputs:
        --None, except for the module itself--

        Returns:
        date (string): The last trading date.
        '''
        return list(self.total.keys())[-1]

    def calc_num_of_days(self, start, end):
        '''
        This function computes the number of calendar days that have passed during the period
        specified.

        Inputs:
        start (date): The date the initial AUM was invested
        end (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.

        Returns:
        days (int): Number of calendar days between the first and last trading days.
        '''
        days = date.fromisoformat(end) - date.fromisoformat(start)
        return days.days

    def get_final_value(self, end):
        '''
        This function returns the final value of the AUM.

        Inputs:
        end (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.

        Returns:
        AUM (float): Final value of the AUM, calculated using the number of initial shares bought
        multiplied by the closing price on the final trading day.
        '''
        return self.total[end][0]

    def calc_tsr(self, start, end, tick):
        '''
        This function calculates the total stock return using data from the large dictionary.
        This value tells the user how lucrative the stock has been taking into account the
        dividents recieved throughout the specified period.

        Inputs:
        start (date): The date the initial AUM was invested
        end (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.
        tick (string): 4 character ticker name, as provided by the user

        Returns:
        total_stock_return (float): Calculated by taking the final closing price of the stock
        summed with the dividends received throughout the period, and dividing the obtained value
        by the intial stock closing price, and finally subtracting the result by 1.
        '''
        initial_price = self.tick_data.trading_data[start][tick][0]
        final_price = self.tick_data.trading_data[end][tick][0]
        divs = sum([x[tick][2] for x in list(self.tick_data.trading_data.values())])
        tsr = (final_price+divs)/initial_price - 1
        return tsr

    def calc_tr(self, start, end):
        '''
        This function calculates the total return using data from the large dictionary.
        This value tells the user how lucrative their portfolio has been taking into account
        the starting and final values of their AUM. If more stocks were included in the users
        portfolio or if fractional stocks were not possible, this value would be different
        from the Profit and Loss value of the stock that is also generated later on.

        Inputs:
        start (date): The date the initial AUM was invested
        end (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.

        Returns:
        total_return (float): Calculated by taking the final value of the AUM and subtracting
        the initial value of the AUM, and finally dividing the obtained value by the initial
        value of the AUM again.
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
        total_return (float): The total return of the portfolio as calculated earlier.

        Returns:
        annualized rate of return (float): Calculated by taking the time that has passed as a
        percentage of a hypothetical trading year and raising the result as the power of the
        total stock return subtracted by 1
        '''
        years = len(self.total) / 250
        aror = (total_return + 1)**(1/years) - 1
        return aror

    def calc_avg_aum(self):
        '''
        This function calculates the average value of the AUM throughout the period.

        Inputs:
        --None, except for the module itself--

        Returns:
        average AUM (float): The mean value of the AUM.
        '''
        return np.mean([x[0] for x in list(self.total.values())])

    def get_max_aum(self):
        '''
        This function calculates the maximum value of the AUM throughout the period.

        Inputs:
        --None, except for the module itself--

        Returns:
        max AUM (float): The maximum value of the AUM.
        '''
        return max([x[0] for x in list(self.total.values())])

    def get_pnl(self, start, end):
        '''
        This function lets the user know how much profit or loss has been made throughout
        their investment period. The formula used is identical to the total_return function
        earlier since the portfolio only consists of 1 stock with fractional shares.

        Inputs:
        start (date): The date the initial AUM was invested
        end (date): The final date to be used for calculations. If no ending date is
        provided by the user, the script takes the latest possible date.


        Returns:
        profit_loss (float): Calculated by taking the final value of the AUM and subtracting
        the initial value of the AUM, and finally dividing the obtained value by the initial
        value of the AUM again.
        '''
        initial_aum = self.total[start][0]
        end_aum = self.total[end][0]
        pnl = (end_aum - initial_aum)/initial_aum
        return pnl

    def get_adr(self, total_return):
        '''
        This function calculates the average daily return of the stock.

        Inputs:
        total_return (float): The total return of the portfolio as calculated earlier.

        Returns:
        adr (float): Calculated by taking the total stock return and dividing the value by the
        total number of trading days.
        '''
        adr = total_return / len(self.total)
        return adr

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
        return np.std([x[1] for x in list(self.total.values()) if x[1] != 0])

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
        sharpe_ratio = (adr-risk_free)/std
        return sharpe_ratio
    def get_plot_daily_aum(self):
        '''
        This function generates the plot of the value of the AUM across the trading period.

        Inputs:
        --None, except for the module itself--

        Returns:
        the plot (jpg): The plot is saved to the user's local drive.
        If the user wishes to run the script again using a different ticker, the new
        plot will override the previous one under the same name.
        '''
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 15))
        plt.xlabel("Date")
        plt.ylabel("AUM")
        plt.title("AUM Across Time")
        aum_vals = pd.DataFrame.from_dict(self.total, orient = 'index')
        aum_vals = aum_vals.reset_index()
        aum_vals["index"] = aum_vals["index"].astype("datetime64")
        aum_vals = aum_vals.set_index("index")
        plt.plot(aum_vals[0])
        plt.xticks(rotation=45)
        plt.savefig("AUM.jpg")
        plt.show()
    def get_plot_ic(self):
        '''
        This function generates the plot of the monthly cumulative IC (Information Coefficient)

        Inputs:
        --None, except for the module itself--

        Returns:
        the plot (jpg): The plot is saved to the user's local drive.
        If the user wishes to run the script again using a different ticker, the new
        plot will override the previous one under the same name.
        '''
        plt.figure(figsize=(15, 15))
        plt.xlabel("Date")
        plt.ylabel("Cumm IC")
        plt.title("Cumm IC Across Time")
        info_vals = pd.DataFrame.from_dict(self.ic_values, orient = 'index')
        info_vals = info_vals.reset_index()
        info_vals["index"] = info_vals["index"].astype("datetime64")
        info_vals = info_vals.set_index("index")
        plt.plot(info_vals[1])
        plt.xticks(rotation=45)
        plt.savefig("Cumulative IC.jpg")
        plt.show()
def main (tickers, b_date, e_date, initial_aum, strategy_type, days, top_pct):
    '''
    This is the main function for initializing, computing, and retrieving the final output.
    The printed outputs are also generated here, color coded according to their negative/positive
    values.

    Inputs:
    tickers (string): A list of 4 character ticker name strings, separated by commas
    b_date (string): The date the initial AUM was invested
    e_date (string): The final date to be used for calculations. If no ending date is
    provided by the user, the script takes the latest possible date
    initial_aum (integer): The intial amount of assets that are invested in the stock
    strategy_type (string): The backtesting strategy chosen by the user
    days (int): the numbers of trading days used to compute strategy-related returns
    top_pct (int): Apositive integer from 1 to 100 that
    indicates the percentage of stocks to pick to go long

    Returns:
    stock (class): The module containing all the relevant information about the stock.
    '''
    if days > 250:
        raise ValueError('--days exceeds 250 days.')
    if days < 1:
        raise ValueError('--days is not a positive number.')
    if top_pct < 1 or top_pct > 100:
        raise ValueError('The integer for the percentage of stocks needs to be between 1 and 100.')
    if strategy_type != 'M' and strategy_type != 'R':
        raise ValueError('There is no such strategy type.')
    tickers_list = tickers.split(',')
    for ticker in tickers_list:
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
    portfolio = Portfolio(tickers_list, beginning_date, ending_date,
    initial_aum,strategy_type, days, top_pct)
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

    print('1 (Begin Date):', beginning_date)
    print('2 (End Date):', ending_date)
    print('3 (Number of Days):', num_days)
    print('4 Total stock return -')
    for tick in tickers_list:
        print(tick + ':', portfolio.calc_tsr(beginning_date, ending_date, tick))

    print('5 (Total return):', total_return)
    print('6 (Annualized RoR):',aror)
    print('7 (Initial AUM):', initial_aum)
    print('8 (Final AUM):', final_aum)
    print('9 (Avg AUM):', average_aum)
    print('10 (Max AUM):', max_aum)
    print('11 (PnL AUM):', pnl)
    print('12 (Avg daily return):', adr)
    print('13 (Daily Std Dev):', std)
    print('14 (Daily Sharpe Ratio):', sharpe)

    portfolio.get_plot_daily_aum()
    portfolio.get_plot_ic()

    return portfolio
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tickers', required=True, help="tickers of the stocks")
    parser.add_argument('--b', type=int, required=True, help="the beginning date of the period")
    parser.add_argument('--e', type=int, help="the ending date of the period")
    parser.add_argument('--initial_aum', type=int,required=True,\
         help="initial asset under management")
    parser.add_argument('--strategy_type', required=True,\
         help="either 'M'(momentum) or 'R'(reversal)")
    parser.add_argument('--days', type=int, required=True,\
         help="the numbers of trading days used to compute strategy-related returns")
    parser.add_argument('--top_pct', type=int,required=True,\
         help="an integer from 1 to 100, the percentage of stocks to pick to go long")
    args=parser.parse_args()
    main(args.tickers, args.b, args.e, args.initial_aum,\
         args.strategy_type, args.days, args.top_pct)
