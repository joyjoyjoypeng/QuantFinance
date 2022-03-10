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
    This function tests whether the Stock() class stores, computer, and retrieves data correctly

    Inputs:

    Returns:
    '''
    beginning_date = date(2021, 3, 1)
    ending_date = date(2022,3,6)
    stock = get_prices.Stock("MSFT", beginning_date, ending_date, 100000)
    # tsr = stock.calc_tsr()
    # tr = stock.calc_tr(tsr)
    # aror = stock.calc_aror(tsr)
    # avg_aum = stock.calc_avg_aum()
    # max_aum = stock.get_max_aum()
    # pnl = stock.get_pnl(100000)
    # avg_return = stock.
    # sd = stock.
    # sharpe_ratio = stock.
    assert stock.beginning_date == '2021-03-01'
    assert stock.ending_date == '2022-03-04'
    assert stock.initial_aum == 100000
    assert stock.num_days == 370
    # assert tsr ==
    # assert tr ==
    # assert aror ==
    # assert avg_aum ==
    # assert max_aum ==
    # assert pnl ==
    # assert avg_dreturn ==
    # assert sd ==
    # assert sharpe_ratio ==

def test_main_return():
    '''
    This function tests whether the main() class stores, computer, and retrieves data correctly
    by comparing against the output of the Stock() class

    Inputs:

    Returns:
    '''
    stock_main = get_prices.main("MSFT", 20210301, 20220306, 100000)
    # tsr_main = stock_main.calc_tsr()
    # tr_main = stock_main.calc_tr(tsr)
    # aror_main = stock_main.calc_aror(tsr)
    # avg_aum_main = stock_main.calc_avg_aum()
    # max_aum_main = stock_main.get_max_aum()
    # pnl_main = stock_main.get_pnl(100000)
    # avg_return_main = stock_main.
    # sd_main = stock_main.
    # sharpe_ratio_main = stock_main.

    stock_class = get_prices.Stock("MSFT", date(2021, 3, 1), date(2022,3,6), 100000)
    # tsr_class = stock_class.calc_tsr()
    # tr_class = stock_class.calc_tr(tsr)
    # aror_class = stock_class.calc_aror(tsr)
    # avg_aum_class = stock_class.calc_avg_aum()
    # max_aum_class = stock_class.get_max_aum()
    # pnl_class = stock_class.get_pnl(100000)
    # avg_return_class = stock_class.
    # sd_class = stock_class.
    # sharpe_ratio_class = stock_class.

    assert stock_main.beginning_date == stock_class.beginning_date
    assert stock_main.ending_date == stock_class.ending_date
    assert stock_main.initial_aum == stock_class.initial_aum
    assert stock_main.num_days == stock_class.num_days

    # assert tsr_main == tsr_class
    # assert tr_main == tr_class
    # assert aror_main == aror_class
    # assert avg_aum_main == avg_aum_class
    # assert max_aum_main == max_aum_class
    # assert pnl_main == pnl_class
    # assert avg_return_main == avg_return_class
    # assert sd_main == sd_class
    # assert sharpe_ratio_main == sharpe_ratio_class

def test_main_no_end_date():
    '''
    This function tests whether the main() class stores, computer, and retrieves data correctly
    given that the optional ending date argument is not present

    Inputs:

    Returns:
    '''
    stock_main = get_prices.main("MSFT", 20220101, None, 100000)
    # tsr_main = stock_main.calc_tsr()
    # tr_main = stock_main.calc_tr(tsr)
    # aror_main = stock_main.calc_aror(tsr)
    # avg_aum_main = stock_main.calc_avg_aum()
    # max_aum_main = stock_main.get_max_aum()
    # pnl_main = stock_main.get_pnl(100000)
    # avg_return_main = stock_main.
    # sd_main = stock_main.
    # sharpe_ratio_main = stock_main.

    stock_today = get_prices.Stock("MSFT", date(2022, 1, 1), date.today() ,100000)
    # tsr_today = stock_today.calc_tsr()
    # tr_today = stock_today.calc_tr(tsr)
    # aror_today = stock_today.calc_aror(tsr)
    # avg_aum_today = stock_today.calc_avg_aum()
    # max_aum_today = stock_today.get_max_aum()
    # pnl_today = stock_today.get_pnl(100000)
    # avg_return_today = stock_today.
    # sd_today = stock_today.
    # sharpe_ratio_today = stock_today.

    assert stock_main.beginning_date == stock_today.beginning_date
    assert stock_main.ending_date == stock_today.ending_date
    assert stock_main.initial_aum == stock_today.initial_aum
    assert stock_main.num_days == stock_today.num_days

    # assert tsr_main == tsr_today
    # assert tr_main == tr_today
    # assert aror_main == aror_today
    # assert avg_aum_main == avg_aum_today
    # assert max_aum_main == max_aum_today
    # assert pnl_main == pnl_today
    # assert avg_return_main == avg_return_today
    # assert sd_main == sd_today
    # assert sharpe_ratio_main == sharpe_ratio_today
