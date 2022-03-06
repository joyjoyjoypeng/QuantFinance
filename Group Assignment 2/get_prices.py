'''
Author: Vasu Namdea & Peng Chiao-Yin (Joy)
Date: 13 Mar 2022

This program fetches daily close prices from the internet for a given ticker and time frame,
and computes some analytics on them.

The program can be called using:
get_prices.py --ticker <xxx> –-b <YYYYMMDD> –-e <YYYYMMMDD> --initial_aum <yyyy>  --plot
'''

import csv
import yfinance as yf
from datetime import date
import numpy as np