# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
"""
# run this  !pip install pandas_datareader
from pandas_datareader import data as web
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday_name  
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

# ticker='MSFT'
# start_date='2014-01-01'
# end_date='2018-12-31'
# s_window = 14
# l_window = 50
# input_dir = r'C:\Users\epinsky\bu\python\data_science_with_Python\datasets'
# output_file = os.path.join(input_dir, ticker + '.csv')
#
# df = get_stock(ticker, start_date, end_date, s_window, l_window)
# df.to_csv(output_file, index=False)
#
#
# with open(output_file) as f:
#     lines = f.read().splitlines()
#
#
#
# df[['Short_MA', 'Long_MA', 'Adj Close']].plot()
#
#
# #df_2 = pd.read_csv(output_file)
#
#
#
#
# x = 1
#














