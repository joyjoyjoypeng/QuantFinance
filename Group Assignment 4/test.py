'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 14 Apr 2022

This is a test file for backtest_strategy.py
'''

from datetime import date
import backtest_two_signal_strategy
import pytest
def test_main_argument():
    '''
    This function tests whether invalid arguments raise an error
    '''
    with pytest.raises(TypeError):
        backtest_two_signal_strategy.main('AAPL,TSLA', 20210112, 20220412, "1000",\
        "M", 100, "R", 100, 70)
    with pytest.raises(NameError):
        backtest_two_signal_strategy.main('Fake,TSLA', 20210112, 20220412, 1000,\
        "M", 100, "R", 100, 70)
    with pytest.raises(ValueError):
        backtest_two_signal_strategy.main('AAPL,TSLA', 20220112, 20210412, 1000,\
        "M", 100, "R", 100, 70)
    with pytest.raises(ValueError):
        backtest_two_signal_strategy.main('AAPL,TSLA', 2021010, 20220412, 1000,\
        "M", 100, "R", 100, 70)
    with pytest.raises(ValueError):
        backtest_two_signal_strategy.main('AAPL,TSLA', 20210112, 20220412, 1000,\
        "M", 270, "R", 100, 70)
    with pytest.raises(ValueError):
        backtest_two_signal_strategy.main('AAPL,TSLA', 20210112, 20220412, 1000,\
        "M", 100, "R", -1, 70)


def test_portfolio_class():
    '''
    This function tests whether the Portfolio() class stores, computer,
    and retrieves data correctly.
    '''
    tickers = 'AAPL,TSLA,MSFT,FB'
    tickers_list = tickers.split(',')
    portfolio = backtest_two_signal_strategy.Portfolio(tickers_list, date(2021,1,12),
    date(2022, 5, 4), 1000, "M", 100, "R", 100, 25)
    assert portfolio.beginning_date == '2021-01-12'
    assert portfolio.ending_date == '2022-05-03'
    assert portfolio.calc_num_of_days(portfolio.beginning_date, portfolio.ending_date) == 476
    total_return = portfolio.calc_tr(portfolio.beginning_date, portfolio.ending_date)
    assert round(total_return,4) == -0.4627
    assert round(portfolio.calc_aror(total_return),4) == -0.3754
    assert round(portfolio.get_final_value(portfolio.ending_date),2) == 537.26
    assert round(portfolio.calc_avg_aum(),2) == 806.43
    assert round(portfolio.get_max_aum(),2) == 1099.88
    assert round(portfolio.get_pnl(portfolio.beginning_date, portfolio.ending_date),4) == -0.4627
    adr = portfolio.get_adr(total_return)
    assert round(adr,4) == -0.0014
    std = portfolio.get_std()
    assert round(std,4) == 0.0334
    sharpe = portfolio.get_sharpe(adr, std)
    assert round(sharpe,5) == -0.04498

def test_main_no_end_date():
    '''
    This function tests whether the main() class stores, computer, and retrieves data correctly
    given that the optional ending date argument is not present
    '''
    tickers = 'AAPL,TSLA,MSFT,FB'
    tickers_list = tickers.split(',')
    portfolio_main = backtest_two_signal_strategy.main('AAPL,TSLA,MSFT,FB', 20210112,
    None, 1000, "M", 100, "R", 100, 25)
    portfolio_today = backtest_two_signal_strategy.Portfolio(tickers_list, date(2021,1,12),
    date.today(), 1000, "M", 100, "R", 100, 25)
    assert portfolio_main.beginning_date == portfolio_today.beginning_date
    assert portfolio_main.ending_date == portfolio_today.ending_date
    assert portfolio_main.calc_num_of_days(portfolio_main.beginning_date,
    portfolio_main.ending_date) == portfolio_today.calc_num_of_days(portfolio_today.beginning_date,
    portfolio_today.ending_date)
    tr_main = portfolio_main.calc_tr(portfolio_main.beginning_date, portfolio_main.ending_date)
    tr_today = portfolio_today.calc_tr(portfolio_today.beginning_date, portfolio_today.ending_date)
    assert round(tr_main,4) == round(tr_today,4)
    assert round(portfolio_main.calc_aror(tr_main),4)==round(portfolio_today.calc_aror(tr_today),4)
    assert round(portfolio_main.get_final_value(portfolio_main.ending_date),3) == round(
        portfolio_today.get_final_value(portfolio_today.ending_date),3)
    assert round(portfolio_main.calc_avg_aum(),4) == round(portfolio_today.calc_avg_aum(),4)
    assert round(portfolio_main.get_max_aum(),4) == round(portfolio_today.get_max_aum(),4)
    assert round(portfolio_main.get_pnl(portfolio_main.beginning_date,portfolio_main.ending_date),
    4)==round(portfolio_today.get_pnl(portfolio_today.beginning_date,portfolio_today.ending_date),4)
    adr_main = portfolio_main.get_adr(tr_main)
    adr_today = portfolio_today.get_adr(tr_today)
    assert round(adr_main,6) == round(adr_today,6)
    std_main = portfolio_main.get_std()
    std_today = portfolio_today.get_std()
    assert round(std_main,4) == round(std_today,4)
    sharpe_main = portfolio_main.get_sharpe(adr_main, std_main)
    sharpe_today = portfolio_today.get_sharpe(adr_today, std_today)
    assert round(sharpe_main,4) == round(sharpe_today,4)
