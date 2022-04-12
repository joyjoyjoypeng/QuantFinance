'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 14 Apr 2022

This program fetches daily close prices from the internet for given tickers and time frame,
and then back tests some simple momentum and reversal monthly strategies.

The program can be called using:
backtest_strategy.py --tickers <ttt> –-b <YYYYMMDD> –-e <YYYYMMMDD> --initial_aum <yyyy> 
--strategy_type <xx> --days <xx> --top_pct <xx>
'''
import argparse
from datetime import date, timedelta
import numpy as np
import yfinance as yf
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd

class Tickers_data:
    def __init__(self, tickers_list, beginning_date, ending_date, aum, strategy_type, days, top_pct):
        self.giant=self.compute_giant_date(tickers_list, beginning_date, ending_date)[0]
        self.dates=self.compute_giant_date(tickers_list, beginning_date, ending_date)[1]
        self.strnum = ceil(len(tickers_list) * (top_pct/100))
        self.strat=self.compute_strategy(strategy_type, days)
        self.mm=self.compute_mm(tickers_list, beginning_date, ending_date, aum)[0]
    def compute_giant_date(self, tickers_list, beginning_date, ending_date):
        giant = {}
        dates = []
        comp = 'start'

        for ticker in tickers_list:
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(start = beginning_date-timedelta(450), end = ending_date)
            giant[ticker] = data
            if len(dates) == 0:
                for x in range(len(data['Close'])):
                    y = data['Close'].index[x].date()
                    if comp == 'start':
                        comp = y
                    elif comp.month != y.month:
                        if comp >= beginning_date and comp <= ending_date:
                            dates.append(comp)
                    comp = y
        return (giant, dates)
    def compute_strategy(self, strategy_type, days):
        strat = {}

        for dd in self.dates:
            strat[dd] = {}
            for tick, data in self.giant.items():
                for x in range(len(data['Close'])):
                    y = data['Close'].index[x].date()
                    if y == dd:  
                        if strategy_type == 'R':
                            endp = data['Close'][x]
                            sttp = data['Close'][x-days]
                        else:
                            endp = data['Close'][x-20]
                            sttp = data['Close'][x-20-days]
                        ret = (endp - sttp) / sttp
                        strat[dd][tick] = ret
                        
        strat_ticks = {}

        for dd, stats in strat.items():
            strat_ticks[dd] = []
            ranked = dict(sorted(stats.items(), key=lambda item: item[1]))
            num = 0
            while num != self.strnum:
                strat_ticks[dd].append(list(ranked.keys())[num])
                num += 1

        return strat_ticks
    def compute_mm(self, tickers_list, beginning_date, ending_date, aum):
        mm = {}
        ic = {}

        for x in range(len(self.giant[tickers_list[0]]['Close'])):
            y = self.giant[tickers_list[0]]['Close'].index[x].date()
            if y >= beginning_date and y <= ending_date:
                if y in self.dates:
                    if len(mm) == 0:
                        mm[str(y)] = {}
                        for z in tickers_list:
                            if z in self.strat[y]:
                                shares = (aum/self.strnum) / self.giant[z]['Close'][x]
                                mm[str(y)][z] = [self.giant[z]['Close'][x], (aum/self.strnum), shares, self.giant[z]['Dividends'][x]]
                            else:
                                mm[str(y)][z] = [self.giant[z]['Close'][x], 0, 0, self.giant[z]['Dividends'][x]]
                    else:
                        mm[str(y)] = {}
                        prev = self.giant[tickers_list[0]]['Close'].index[x-1].date()
                        new_aum = 0
                        for z in tickers_list:
                            new_aum += mm[str(prev)][z][2] * self.giant[z]['Close'][x]
                        if new_aum == 0:
                            new_aum += aum
                        
                        for z in tickers_list:
                            if z in self.strat[y]:
                                shares = (new_aum/self.strnum) / self.giant[z]['Close'][x]
                                mm[str(y)][z] = [self.giant[z]['Close'][x], (new_aum/self.strnum), shares, self.giant[z]['Dividends'][x]]
                            else:
                                mm[str(y)][z] = [self.giant[z]['Close'][x], 0, 0, self.giant[z]['Dividends'][x]]
                        
                        pos = self.dates.index(y)
                        if pos != 0:
                            ic[y] = 0
                            prem = self.dates[pos-1]
                            for z in tickers_list:
                                if z in self.strat[prem]:
                                    if mm[str(y)][z][0] > mm[str(prem)][z][0]:
                                        ic[y] += 1
                            ic[y] = [2 * (ic[y]/self.strnum) - 1]
                                
                else:
                    if len(mm) == 0:
                        mm[str(y)] = {}
                        for z in tickers_list:
                            mm[str(y)][z] = [self.giant[z]['Close'][x], 0, 0, self.giant[z]['Dividends'][x]]
                    else:
                        prev = self.giant[tickers_list[0]]['Close'].index[x-1].date()
                        mm[str(y)] = {}
                        for z in tickers_list:
                            shares = mm[str(prev)][z][2]
                            mm[str(y)][z] = [self.giant[z]['Close'][x], shares * self.giant[z]['Close'][x], shares, self.giant[z]['Dividends'][x]]
        tot = 0

        for dd, val in ic.items():
            tot += val[0]
            ic[dd].append(tot)
        return (mm,ic)       
    def compute_total(self, aum):
        total = {}
        place = 0

        for dd, stocks in self.mm.items():
            newaum = 0
            for z in stocks.values():
                newaum += z[1]
            if place != 0:
                dpr = (newaum-place)/place
            else:
                dpr = 0
            if newaum == 0:
                newaum = aum
                dpr = 0
            total[dd] = [newaum, dpr]
            place = newaum
        return total
class Portfolio:
    def __init__(self, tickers_list, beginning_date, ending_date, aum, strategy_type, days, top_pct):
        self.tick_data=Tickers_data(tickers_list, beginning_date, ending_date, aum, strategy_type, days, top_pct)
        self.total=self.tick_data.compute_total(aum)
        self.beginning_date=self.get_start_date()
        self.ending_date=self.get_end_date()
        self.ic=self.tick_data.compute_mm(tickers_list, beginning_date, ending_date, aum)[1]
    def get_start_date(self):
        return list(self.total.keys())[0]

    def get_end_date(self):
        return list(self.total.keys())[-1]

    def calc_num_of_days(self, start, end):
        days = date.fromisoformat(end) - date.fromisoformat(start)
        return days.days

    def get_final_value(self, end):
        return self.total[end][0]

    def calc_TSR(self, start, end, tick):
        initial_price = self.tick_data.mm[start][tick][0]
        final_price = self.tick_data.mm[end][tick][0]
        divs = sum([x[tick][2] for x in list(self.tick_data.mm.values())])
        tsr = (final_price+divs)/initial_price - 1
        return tsr

    def calc_TR(self, start, end):
        initial_aum = self.total[start][0]
        end_aum = self.total[end][0]
        total_return = (end_aum - initial_aum)/initial_aum
        return total_return

    def calc_aror(self, tr):
        years = len(self.total) / 250 
        aror = (tr + 1)**(1/years) - 1
        return aror

    def calc_avg_aum(self):
        return np.mean([x[0] for x in list(self.total.values())])

    def get_max_aum(self):
        return max([x[0] for x in list(self.total.values())])

    def get_pnl(self, start, end):
        initial_aum = self.total[start][0]
        end_aum = self.total[end][0]
        pnl = (end_aum - initial_aum)/initial_aum
        return pnl

    def get_adr(self, tr):
        adr = tr / len(self.total)
        return adr
        
    def get_std(self):
        return np.std([x[1] for x in list(self.total.values()) if x[1] != 0])

    def get_sharpe(self, adr, std, risk_free = 0.0001):
        sr = (adr-risk_free)/std
        return sr
    def get＿plot_daily_AUM(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 15))
        plt.xlabel("Date")
        plt.ylabel("AUM")
        plt.title("AUM Across Time")
        df = pd.DataFrame.from_dict(self.total, orient = 'index')
        df = df.reset_index()
        df["index"] = df["index"].astype("datetime64")
        df = df.set_index("index")
        plt.plot(df[0])
        plt.xticks(rotation=45)
        plt.savefig("AUM.jpg")
    def get＿plot_IC(self):
        plt.figure(figsize=(15, 15))
        plt.xlabel("Date")
        plt.ylabel("Cumm IC")
        plt.title("Cumm IC Across Time")
        df = pd.DataFrame.from_dict(self.ic, orient = 'index')
        df = df.reset_index()
        df["index"] = df["index"].astype("datetime64")
        df = df.set_index("index")
        plt.plot(df[1])
        plt.xticks(rotation=45)
        plt.savefig("Cumulative IC.jpg")
def main (tickers, b_date, e_date, initial_aum, strategy_type, days, top_pct):
    if days > 250:
        raise ValueError('The number of trading days used to compute strategy-related exceeds 250 days.')
    if days < 1:
        raise ValueError('The number of trading days used to compute strategy-related is not a positive number.')
    if top_pct < 1 or top_pct > 100:
        raise ValueError('The integer for the percentage of stocks to pick to go long needs to be between 1 and 100.')
    if strategy_type != 'M' and strategy_type != 'R':
        raise ValueError('There is no such strategy type.')
    tickers_list = tickers.split(',')
    for (ticker_index, ticker) in enumerate(tickers_list):
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
    portfolio = Portfolio(tickers_list, beginning_date, ending_date, initial_aum,strategy_type, days, top_pct)
    beginning_date = portfolio.get_start_date()
    ending_date = portfolio.get_end_date()
    num_days = portfolio.calc_num_of_days(beginning_date, ending_date)
    total_return = portfolio.calc_TR(beginning_date, ending_date)
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
        print(tick + ':', portfolio.calc_TSR(beginning_date, ending_date, tick))

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

    portfolio.get＿plot_daily_AUM()
    portfolio.get＿plot_IC()

    return portfolio
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tickers', required=True, help="tickers of the stocks")
    parser.add_argument('--b', type=int, required=True, help="the beginning date of the period")
    parser.add_argument('--e', type=int, help="the ending date of the period")
    parser.add_argument('--initial_aum', type=int,required=True, help="initial asset under management")
    parser.add_argument('--strategy_type', required=True, help="either 'M'(momentum) or 'R'(reversal)")
    parser.add_argument('--days', type=int, required=True, help="the numbers of trading days used to compute strategy-related returns")
    parser.add_argument('--top_pct', type=int,required=True, help="an integer from 1 to 100, the percentage of stocks to pick to go long")
    args=parser.parse_args()
    main(args.tickers, args.b, args.e, args.initial_aum, args.strategy_type, args.days, args.top_pct)
