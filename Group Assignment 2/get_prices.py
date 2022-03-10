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
'''
Class Stock
'''
class Stock:
    def __init__(self, ticker, beginning_date, ending_date, aum):
        self.tick_data=self.compute_mm(ticker, beginning_date, ending_date, aum)
        self.beginning_date=self.get_beginning_date()
        self.ending_date=self.get_end_date()
        self.num_days = (ending_date-beginning_date).days
        self.initial_aum=aum

    #need to change the for loop formatting
    def compute_mm(self, ticker, start_d, end_d, aum):
        data = yf.Ticker(ticker).history(period='max')
        mm={}
        for x in range(len(data['Close'])):
            y=date.fromisoformat(str(data['Close'].index[x]).split(' ')[0])
            if (y-start_d).days>=0 and (y-end_d).days<=0:
                if len(mm) == 0:
                    shares=aum // data['Close'][x]
                    invested=shares * data['Close'][x]
                    remainder=aum-invested
                    mm[str(y)]=[data['Close'][x], invested, data['Dividends'][x]]
                else:
                    mm[str(y)]=[data['Close'][x], shares * data['Close'][x], data['Dividends'][x]]
        return mm

    def get_beginning_date(self):
        return list(self.tick_data.keys())[0]

    def get_end_date(self):
        return list(self.tick_data.keys())[-1]

    def get_final_value(self):
        return self.tick_data[self.ending_date][1]

    def calc_TSR(self):
        initial_price = self.tick_data[self.beginning_date][0]
        final_price = self.tick_data[self.ending_date][0]
        divs = sum([x[2] for x in list(self.tick_data.values())])
        tsr = (final_price-initial_price+divs)/initial_price
        return tsr

    def calc_TR(self, tsr):
        return self.initial_aum * tsr

    def calc_aror(self,tsr):
        years = len(self.tick_data) / 250
        aror = (tsr + 1)**(1/years) - 1
        return aror * self.initial_aum

    def calc_avg_aum(self):
        return np.mean([x[1] for x in list(self.tick_data.values())])

    def get_max_aum(self):
        return max([x[1] for x in list(self.tick_data.values())])

    def get_pnl(self,final_aum):
        return (final_aum - self.initial_aum)/self.initial_aum * 100

'''
    This is the main function containing...
'''
def main (ticker, b_date, e_date, initial_aum, plot=None):
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
    tsr = stock.calc_TSR()
    tr = stock.calc_TR(tsr)
    aror= stock.calc_aror(tsr)
    final_aum = stock.get_final_value()
    average_aum = stock.calc_avg_aum()
    max_aum = stock.get_max_aum()
    pnl = stock.get_pnl(final_aum)
    # if plot == true
    #     plot...
    print(stock.beginning_date,"\nEnding date:", stock.ending_date, "\nNumber of days:", stock.num_days,
    "\nTotal stock return:", tsr, "\nTotal return: ", tr, "\nAnnualized rate of return:", aror,
    "\nInitial AUM:",initial_aum, "\nFinal AUM", final_aum, "\nAverage AUM:", average_aum,
    "\nMaximum AUM:", max_aum, "\nPnL:", pnl, "\nAverage daily return of the portfolio:",
    "\nDaily Standard deviation of the return of the portfolio:", "\nDaily Sharpe Ratio of the portfolio:")
    return stock

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True, help="ticker of the stock")
    parser.add_argument('--b',type=int, required=True, help="the beginning date of the period")
    parser.add_argument('--e', type=int, help="the ending date of the period")
    parser.add_argument('--initial_aum', type=int, required=True,help="number of k-folds")
    parser.add_argument('--plot', action='store_true', help="a plot of the input and prediction data")
    args=parser.parse_args()
    main(args.ticker, args.b, args.e, args.initial_aum, args.plot)
