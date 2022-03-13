'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 13 Mar 2022

This is a test file for get_prices.py
'''

from datetime import date
import get_prices
import pytest
def test_main_argument():
    '''
    This function tests whether invalid arguments raise an error

    Inputs:
        -- No inputs are required as this function assumes an error will be raised
        when insufficient arguments are passed to the main function --

    Returns:
        -- Nothing should be returned ideally if only the predicted errors are raised --
    '''
    with pytest.raises(TypeError):
        get_prices.main("MSFT", 20210301, 20220306, "ten thousands")#incorrect AUM format
    with pytest.raises(NameError):
        get_prices.main("FakeTicker", 20210301, 20220306, 100000)
    with pytest.raises(TypeError):
        get_prices.main()                #Non-existing ticker name
    with pytest.raises(TypeError):
        get_prices.main("MSFT", 100000)  #incorrect number of expected arguments
    with pytest.raises(ValueError):
        get_prices.main("MSFT", 2021030, 20220306, 100000)          #incorrect date syntax

def test_stock_class():
    '''
    This function tests whether the Stock() class stores, computer, and retrieves data correctly.
    '''
    beginning_date = date(2021,3,1)
    ending_date = date(2022,3,6)
    stock = get_prices.Stock("MSFT", beginning_date, ending_date, 10000)
    assert stock.ticker_name == "MSFT"
    assert stock.beginning_date == '2021-03-01'
    assert stock.ending_date == '2022-03-04'
    assert stock.calc_days() == 368
    tsr = stock.calc_tsr()
    assert tsr == 0.22854133920617614
    assert stock.calc_tr() == 0.22617241425384327
    assert stock.calc_aror(tsr) == 0.2216731631259674
    assert stock.initial_aum == 10000
    assert stock.get_final_value() == 12261.724142538433
    assert stock.calc_avg_aum() == 12140.62911689067
    assert stock.get_max_aum() == 14514.317958819227
    assert stock.get_pnl() == 0.22617241425384327
    adr = stock.get_adr(tsr)
    assert adr == 0.0008892659113080783
    std = stock.get_std()
    assert std == 0.01432884720813962
    sharpe = stock.get_sharpe(adr, std)
    assert sharpe == 0.055082303540771184

def test_main_return():
    '''
    This function tests whether the main() class stores, computer, and retrieves data correctly
    by comparing against the output of the Stock() class.
    '''
    stock_main = get_prices.main("MSFT", 20210301, 20220306, 10000)
    stock_class = get_prices.Stock("MSFT", date(2021, 3, 1), date(2022,3,6), 10000)
    assert stock_main.ticker_name == stock_class.ticker_name
    assert stock_main.beginning_date == stock_class.beginning_date
    assert stock_main.ending_date == stock_class.ending_date
    assert stock_main.initial_aum == stock_class.initial_aum
    assert stock_main.calc_days() == stock_class.calc_days()
    tsr_main = stock_main.calc_tsr()
    tsr_class = stock_class.calc_tsr()
    assert tsr_main == tsr_class
    assert stock_main.calc_tr() == stock_class.calc_tr()
    assert stock_main.calc_aror(tsr_main) == stock_class.calc_aror(tsr_class)
    assert stock_main.initial_aum == stock_class.initial_aum
    assert stock_main.get_final_value() == stock_class.get_final_value()
    assert stock_main.calc_avg_aum() == stock_class.calc_avg_aum()
    assert stock_main.get_max_aum() == stock_class.get_max_aum()
    assert stock_main.get_pnl() == stock_class.get_pnl()
    adr_main = stock_main.get_adr(tsr_main)
    adr_class = stock_class.get_adr(tsr_class)
    assert adr_main == adr_class
    std_main = stock_main.get_std()
    std_class = stock_class.get_std()
    assert std_main == std_class
    sharpe_main = stock_main.get_sharpe(adr_main, std_main)
    sharpe_class = stock_class.get_sharpe(adr_class, std_class)
    assert sharpe_main == sharpe_class

def test_main_no_end_date():
    '''
    This function tests whether the main() class stores, computer, and retrieves data correctly
    given that the optional ending date argument is not present
    '''
    stock_main = get_prices.main("MSFT", 20220101, None, 10000)
    stock_today = get_prices.Stock("MSFT", date(2022, 1, 1), date.today() ,10000)
    assert stock_main.ticker_name == stock_today.ticker_name
    assert stock_main.beginning_date == stock_today.beginning_date
    assert stock_main.ending_date == stock_today.ending_date
    assert stock_main.initial_aum == stock_today.initial_aum
    assert stock_main.calc_days() == stock_today.calc_days()
    tsr_main = stock_main.calc_tsr()
    tsr_today = stock_today.calc_tsr()
    assert tsr_main == tsr_today
    assert stock_main.calc_tr() == stock_today.calc_tr()
    assert stock_main.calc_aror(tsr_main) == stock_today.calc_aror(tsr_today)
    assert stock_main.initial_aum == stock_today.initial_aum
    assert stock_main.get_final_value() == stock_today.get_final_value()
    assert stock_main.calc_avg_aum() == stock_today.calc_avg_aum()
    assert stock_main.get_max_aum() == stock_today.get_max_aum()
    assert stock_main.get_pnl() == stock_today.get_pnl()
    adr_main = stock_main.get_adr(tsr_main)
    adr_today = stock_today.get_adr(tsr_today)
    assert adr_main == adr_today
    std_main = stock_main.get_std()
    std_today = stock_today.get_std()
    assert std_main == std_today
    sharpe_main = stock_main.get_sharpe(adr_main, std_main)
    sharpe_today = stock_today.get_sharpe(adr_today, std_today)
    assert sharpe_main == sharpe_today
