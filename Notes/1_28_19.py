import numpy as np

# number of stocks that i can purchase:
current_balance = 100  # dollars
adj_stock_price = 60.12345
num_stocks = current_balance / adj_stock_price


def compound_interest(s_balance, rate, t_periods):
    return(s_balance * (1 + rate)**t_periods)

s_balance = 24
years = 400

# in numpy, slices into the data structures are actually views, so modifying the elements of the slice will
# modify the underlying data structure
