# Codsoft task 2

pip install --upgrade pip
/bin/sh: 1: Syntax error: Unterminated quoted string
Note: you may need to restart the kernel to use updated packages.

# Importing required libraries
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import xgboost as xgb
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Import datast 
store = pd.read_csv('../input/rossmann-store-sales/store.csv')
train = pd.read_csv('../input/rossmann-store-sales/train.csv', index_col='Date', parse_dates=True)
test = pd.read_csv('../input/rossmann-store-sales/test.csv')
train.shape, test.shape, store.shape

((1017209, 8), (41088, 8), (1115, 10))

train.head()

	Store	DayOfWeek	Sales	Customers	Open	Promo	StateHoliday	SchoolHoliday
Date								
2015-07-31	1	5	5263	555	1	1	0	1
2015-07-31	2	5	6064	625	1	1	0	1
2015-07-31	3	5	8314	821	1	1	0	1
2015-07-31	4	5	13995	1498	1	1	0	1
2015-07-31	5	5	4822	559	1	1	0	1


test.head()

Id	Store	DayOfWeek	Date	Open	Promo	StateHoliday	SchoolHoliday
0	1	1	4	2015-09-17	1.0	1	0	0
1	2	3	4	2015-09-17	1.0	1	0	0
2	3	7	4	2015-09-17	1.0	1	0	0
3	4	8	4	2015-09-17	1.0	1	0	0
4	5	9	4	2015-09-17	1.0	1	0	0

store.head()

Store	StoreType	Assortment	CompetitionDistance	CompetitionOpenSinceMonth	CompetitionOpenSinceYear	Promo2	Promo2SinceWeek	Promo2SinceYear	PromoInterval
0	1	c	a	        1270.0                   9.0	2008.0	0	NaN	NaN	NaN
1	2	a	a	570.0	11.0	2007.0	1	13.0	2010.0	Jan,Apr,Jul,Oct
2	3	a	a	14130.0	12.0	2006.0	1	14.0	2011.0	Jan,Apr,Jul,Oct
3	4	c	c	620.0	9.0	2009.0	0	NaN	NaN	NaN
4	5	a	a	29910.0	4.0	2015.0	0	NaN	NaN	NaN

1. Explamatory Data Analysis(EDA)
1.1: Trends & Seasonility
How the sales vary with month, promo(First promotional Offer), promo2(Second Promotional Offer) and years.

train.shape
(1017209, 8)
Train data as almost 1M observations of sales data over the year of appriximatelly (2013-2015). Okay, bread Date column in Year, Month, Day, Week columns

# Extract Year, Month, Day, Wee columns 
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekofYear'] = train.index.weekofyear

train['SalesPerCustomer'] = train['Sales']/train['Customers']

train.head()

	Store	DayOfWeek	Sales	Customers	Open	Promo	StateHoliday	SchoolHoliday	Year	Month	Day	WeekofYear	SalesPerCustomer
Date													
2015-07-31	1	5	5263	555	1	1	0	1	2015	7	31	31	9.482883
2015-07-31	2	5	6064	625	1	1	0	1	2015	7	31	31	9.702400
2015-07-31	3	5	8314	821	1	1	0	1	2015	7	31	31	10.126675
2015-07-31	4	5	13995	1498	1	1	0	1	2015	7	31	31	9.342457
2015-07-31	5	5	4822	559	1	1	0	1	2015	7	31	31	8.626118

# Checking the data when the store is closed 
train_store_closed = train[(train.Open == 0)]
train_store_closed.head()

Store	DayOfWeek	Sales	Customers	Open	Promo	StateHoliday	SchoolHoliday	Year	Month	Day	WeekofYear	SalesPerCustomer
Date													
2015-07-31	292	5	0	0	0	1	0	1	2015	7	31	31	NaN
2015-07-31	876	5	0	0	0	1	0	1	2015	7	31	31	NaN
2015-07-30	292	4	0	0	0	1	0	1	2015	7	30	31	NaN
2015-07-30	876	4	0	0	0	1	0	1	2015	7	30	31	NaN
2015-07-29	292	3	0	0	0	1	0	1	2015	7	29	31	NaN

# Check when the store was closed 
train_store_closed.hist('DayOfWeek')
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fc194a84d10>]],
      dtype=object)

From this chart, we could see that, 7th day store was mostly clodes. It is Sunday and makes sense.

# Check whether there school was closed for holyday 
train_store_closed['SchoolHoliday'].value_counts().plot(kind='bar')
<matplotlib.axes._subplots.AxesSubplot at 0x7fc1939bfc90>

Here 1 is school closed day and it pretty low. And 0 is None.
# Check whether there school was closed for holyday 
train_store_closed['StateHoliday'].value_counts().plot(kind='bar')
<matplotlib.axes._subplots.AxesSubplot at 0x7fc1939014d0>


Here, The state is closed for (a= Public holyday, b = Easter holyday, c = Christmas and 0 is None)

# Check the null values
# In here there is no null value 
train.isnull().sum()
Store                    0
DayOfWeek                0
Sales                    0
Customers                0
Open                     0
Promo                    0
StateHoliday             0
SchoolHoliday            0
Year                     0
Month                    0
Day                      0
WeekofYear               0
SalesPerCustomer    172869
dtype: int64

# Number of days with closed stores
train[(train.Open == 0)].shape[0]
172817

# Okay now check No. of dayes store open but sales zero ( It might be caused by external refurbishmnent)
train[(train.Open == 1) & (train.Sales == 0)].shape[0]
54

# Work with store data 
store.head()

Store	StoreType	Assortment	CompetitionDistance	CompetitionOpenSinceMonth	CompetitionOpenSinceYear	Promo2	Promo2SinceWeek	Promo2SinceYear	PromoInterval
0	1	c	a	1270.0	9.0	2008.0	0	NaN	NaN	NaN
1	2	a	a	570.0	11.0	2007.0	1	13.0	2010.0	Jan,Apr,Jul,Oct
2	3	a	a	14130.0	12.0	2006.0	1	14.0	2011.0	Jan,Apr,Jul,Oct
3	4	c	c	620.0	9.0	2009.0	0	NaN	NaN	NaN
4	5	a	a	29910.0	4.0	2015.0	0	NaN	NaN	NaN
# Check null values 
# Most of the columns has null values 

store.isnull().sum()
Store                          0
StoreType                      0
Assortment                     0
CompetitionDistance            3
CompetitionOpenSinceMonth    354
CompetitionOpenSinceYear     354
Promo2                         0
Promo2SinceWeek              544
Promo2SinceYear              544
PromoInterval                544
dtype: int64

# Replacing missing values for Competiton distance with median
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

# No info about other columns - so replcae by 0
store.fillna(0, inplace=True)
# Again check it and now its okay 

store.isnull().sum().sum()
0
# Work with test data 
test.head()

Id	Store	DayOfWeek	Date	Open	Promo	StateHoliday	SchoolHoliday
0	1	1	4	2015-09-17	1.0	1	0	0
1	2	3	4	2015-09-17	1.0	1	0	0
2	3	7	4	2015-09-17	1.0	1	0	0
3	4	8	4	2015-09-17	1.0	1	0	0
4	5	9	4	2015-09-17	1.0	1	0	0
# check null values ( Only one feature Open is empty)
test.isnull().sum()
Id                0
Store             0
DayOfWeek         0
Date              0
Open             11
Promo             0
StateHoliday      0
SchoolHoliday     0
dtype: int64
# Assuming stores open in test
test.fillna(1, inplace=True)
# Again check 
test.isnull().sum().sum()

# Join train and store table 
train_store_joined = pd.merge(train, store, on='Store', how='inner')
train_store_joined.head()

Store	DayOfWeek	Sales	Customers	Open	Promo	StateHoliday	SchoolHoliday	Year	Month	...	SalesPerCustomer	StoreType	Assortment	CompetitionDistance	CompetitionOpenSinceMonth	CompetitionOpenSinceYear	Promo2	Promo2SinceWeek	Promo2SinceYear	PromoInterval
0	1	5	5263	555	1	1	0	1	2015	7	...	9.482883	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
1	1	4	5020	546	1	1	0	1	2015	7	...	9.194139	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
2	1	3	4782	523	1	1	0	1	2015	7	...	9.143403	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
3	1	2	5011	560	1	1	0	1	2015	7	...	8.948214	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
4	1	1	6102	612	1	1	0	1	2015	7	...	9.970588	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0


train_store_joined.groupby('StoreType')['Customers', 'Sales', 'SalesPerCustomer'].sum().sort_values('Sales', ascending='desc')
Customers	Sales	SalesPerCustomer
StoreType			
b	31465621	159231395	7.987612e+04
c	92129705	783221426	9.744876e+05
d	156904995	1765392943	2.918350e+06
a	363541434	3165334859	4.043129e+06
# Closed and zero-sales observations 
train_store_joined[(train_store_joined.Open == 0) | (train_store_joined.Sales==0)].shape
(172871, 22)
So, we have 172,871 observations when the stores were closed or have zero sales.

# Open & Sales >0 stores
train_store_joined_open = train_store_joined[~((train_store_joined.Open ==0) | (train_store_joined.Sales==0))]
train_store_joined_open

	Store	DayOfWeek	Sales	Customers	Open	Promo	StateHoliday	SchoolHoliday	Year	Month	...	SalesPerCustomer	StoreType	Assortment	CompetitionDistance	CompetitionOpenSinceMonth	CompetitionOpenSinceYear	Promo2	Promo2SinceWeek	Promo2SinceYear	PromoInterval
0	1	5	5263	555	1	1	0	1	2015	7	...	9.482883	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
1	1	4	5020	546	1	1	0	1	2015	7	...	9.194139	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
2	1	3	4782	523	1	1	0	1	2015	7	...	9.143403	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
3	1	2	5011	560	1	1	0	1	2015	7	...	8.948214	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
4	1	1	6102	612	1	1	0	1	2015	7	...	9.970588	c	a	1270.0	9.0	2008.0	0	0.0	0.0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1017202	1115	1	6905	471	1	1	0	1	2013	1	...	14.660297	d	c	5350.0	0.0	0.0	1	22.0	2012.0	Mar,Jun,Sept,Dec
1017204	1115	6	4771	339	1	0	0	1	2013	1	...	14.073746	d	c	5350.0	0.0	0.0	1	22.0	2012.0	Mar,Jun,Sept,Dec
1017205	1115	5	4540	326	1	0	0	1	2013	1	...	13.926380	d	c	5350.0	0.0	0.0	1	22.0	2012.0	Mar,Jun,Sept,Dec
1017206	1115	4	4297	300	1	0	0	1	2013	1	...	14.323333	d	c	5350.0	0.0	0.0	1	22.0	2012.0	Mar,Jun,Sept,Dec
1017207	1115	3	3697	305	1	0	0	1	2013	1	...	12.121311	d	c	5350.0	0.0	0.0	1	22.0	2012.0	Mar,Jun,Sept,Dec
