"""

use cumulative product to calculate the cumulative return on a stock

x = np.array([2, 5, 1, 3]
cum_prod = np.cumprod(x)
# array([ 2, 10, 10, 30], dtype=int32)

Example:
    D1: r1  0.07
    D2: r2  -0.05
    D3: 43  0.005

daily_returns = np.array([0.07, -0.05, 0.002, -0.03])
what's the return over four days?  "can't do the sum
formula after 1 year = x + xr1 = x(1 + r1)(1 + r2)(1 + r3)...(1 + rn)

so it becomes:
X = 1 + daily_returns
compounded_returns = np.cumprod(X)

****** Homework Assignment 4 ******
Task 1: Add naive strategy
if return today is > 0, assume the same tomorrow
if return today is < 0, assume it will be the same tomorrow

"good" day is adj_close > 0
"bad" day is adj_close < 0
buy on first good day.  stay, stay stay stay through "good" days
sell (assuming you have stocks) on first "bad" day.

Task 2: generate and plot your label data

For 2018 you labeled your dataset (green and red) for each week
52 labeled weeks for 2018

a) add more labels (at least for 2017)
b) plot your labels as follows

For each week, compute the weekly return and weekly standard deviation

ex: week 15 -----> (R15, std_dev15)

Fri mo tu wed thu fri
100               105 # adj_close

SCRAP THIS, LOOK BELOW
Therefore R15 = .05 (5%) for percentage return.  just "5" for total return.  stick with total return because it's easier
to calculate the std_dev below.
std_dev is std dev of the 5 values from mon to friday of week 15???

given r1, ...., r5 - daily returns for the week
r1 - from prev Friday to mon
r2 - mon - tue
r5 from thu to fri
std_dev = np.std(return_vector for 1 week)

for each week, compute (x, y)
x - mean of 5 daily returns
y - std dev of 5 daily returns

A = np.array([r1, r2, r3, r4, r5])
X = np.mean(A)   # average daily return
Y = np.std(A)

For each week, compute weekly return (Fri to Fri) and plot it together with your label on a graph
X is average daily return over a week, Y is std of daily returns per week, and radius is the weekly return

Do this for 2018 only
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal - length', 'sepal - width', 'petal - length', 'petal - width', 'Class'])

x = np.arange(1, 37).reshape(6, 6)

data_dict = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8],
    'Label': ['green', 'green', 'green', 'green', 'red', 'red', 'red', 'red'],
    'Height': [5, 5.5, 5.33, 5.75,
               6.00, 5.92, 5.58, 5.92],
    'Weight': [100, 150, 130, 150,
               180, 190, 170, 165],
    'Foot': [6, 8, 7, 9, 13, 11, 12, 10]
}
data = pd.DataFrame(data_dict, columns=list(data_dict.keys()))

print(data[['Height', 'Weight']].values)
# Out[31]:
# array([[  5.  , 100.  ],
#        [  5.5 , 150.  ],
#        [  5.33, 130.  ],
#        [  5.75, 150.  ],
#        [  6.  , 180.  ],
#        [  5.92, 190.  ],
#        [  5.58, 170.  ],
#        [  5.92, 165.  ]])

# Use StandardScalar to standardize (using mean/std) of a matrix.  See slides for examples
