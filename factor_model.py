# %%
import pandas as pd
import numpy as np
import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import seaborn as sns

import pyreadr

from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

import torch
from torch import nn
from torch.optim import SGD


# %%
Rdata = pyreadr.read_r('D:/temp/data_ml.RData')

data = Rdata['data_ml']
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')

# %%
features = data.columns.to_list()[2:-4]
features_short = ["Div_Yld", "Eps", "Mkt_Cap_12M_Usd", "Mom_11M_Usd", "Ocf", "Pb", "Vol1Y_Usd"]

# this creates dummy variables indicated whether the return of a given stock was higher
# than the median cross-section return. this will be used as the Y variable in
# categorical prediction problems afterwards. i.e. we'll try to predict which stocks will perform relatively better
data['R1M_Usd_C'] = data.groupby('date')['R1M_Usd'].apply(lambda x: (x > x.median()))
data['R12M_Usd_C'] = data.groupby('date')['R12M_Usd'].apply(lambda x: (x > x.median()))


# %%
separation_date = '2014-1-15'
separation_mask = (data['date'] < separation_date)

training_sample = data.loc[separation_mask]
testing_sample = data.loc[~separation_mask]

# %%
stock_ids = data['stock_id'].unique().tolist()

max_dates = data.groupby('stock_id')['date'].count().max()
stocks_with_max_dates = data.groupby('stock_id')['date'].count() == max_dates
# these are stocks who have data for all timestamps
stock_ids_short = stocks_with_max_dates.where(stocks_with_max_dates).dropna().index.tolist()

returns = data[data['stock_id'].isin(stock_ids_short)][['date', 'stock_id', 'R1M_Usd']]
returns = returns.pivot(index='date', columns='stock_id')

# %%
# we create a simple size factor where a stock is "large" if it's larger than the cross-section median stock
# and small if not. one would expect based on the size factor framework that small stocks outperform large
# stocks.

data['size'] = data.groupby('date')['Mkt_Cap_12M_Usd'].apply(lambda x: (x > x.median())).replace(
    {True: 'Large', False: 'Small'})
data['year'] = data['date'].dt.year

return_by_size = data.groupby(['year', 'size'])['R1M_Usd'].mean().reset_index()

ax = sns.barplot(x='year', y='R1M_Usd', hue='size', data=return_by_size)
ax.set(xlabel='', ylabel='Average return')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()

# %%

# below we download the factor data made available by ken french (yeah, fama's buddy)
# his website is a goldmine of factor-related data, it's highly advisable to do some digging

ff_factors = pd.read_csv('D:/temp/F-F_Research_Data_5_Factors_2x3.csv', skiprows=3)
ff_factors.rename({'Unnamed: 0': 'date', 'Mkt-RF': 'MKT_RF'}, axis=1, inplace=True)
ff_factors = ff_factors.loc[ff_factors['date'].str.strip(' ').str.len() == 6]


def last_day_of_month(date_value):
    return date_value.replace(day=monthrange(date_value.year, date_value.month)[1])


ff_factors['date'] = pd.to_datetime(ff_factors['date'], format='%Y%m').apply(last_day_of_month)
ff_factors[ff_factors.columns[1:]] = ff_factors[ff_factors.columns[1:]].apply(pd.to_numeric) / 100

# %%
# replicating this from the book for completeness only, but i think it's a pretty messy chart
# it's hard to take much insight from it
temp_factors = ff_factors.copy()

temp_factors['date'] = temp_factors['date'].dt.year
temp_factors = pd.melt(temp_factors, id_vars='date')
temp_factors = temp_factors.groupby(['date', 'variable']).mean().reset_index()

ax = sns.lineplot(x='date', y='value', hue='variable', data=temp_factors)
ax.legend(bbox_to_anchor=(1.05, 0.7), loc=2, borderaxespad=0.)
ax.set_title('Average returns over time of common factors')
plt.show()

# %%
# let's see how factors cumulative performance over time
# but wrap that in a function that allows you to choose the start period (as that influences cumulative performance a lot)


def plot_cumulative_performance(df, start_date = None):

  # this function will plot cumulative performance for any wide dataframe of returns (e.g. index is date, columns are assets/factor)
  # optional: you can pass the start date in %m/%d/%y format e.g. '1/1/1995', '12/15/2000'
  # if you don't pass a start date, it will use the whole sample

  cumul_returns = (1+df.set_index('date')).cumprod()

  if start_date is None:
    start_date = cumul_returns.index.min()
  else:
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    cumul_returns = cumul_returns.loc[cumul_returns.index >= start_date]

  first_line = pd.DataFrame([[1. for col in cumul_returns.columns]],
                            columns=cumul_returns.columns,
                            index=[start_date - relativedelta(months=1)])

  cumul_returns = pd.concat([first_line, cumul_returns])

  return cumul_returns.plot(title = f'Cumulative factor performance since {start_date.strftime("%B %Y")}')


plot_cumulative_performance(ff_factors)
plt.show()

# %%
# merging and cleaning up the data before we run the regressions
data_fm = data[['date', 'stock_id', 'R1M_Usd']][data['stock_id'].isin(stock_ids_short)]
data_fm = data_fm.merge(ff_factors, on='date')
data_fm['R1M_Usd'] = data_fm.groupby('stock_id')['R1M_Usd'].shift(1)
data_fm.dropna(inplace=True)

# running time series regressions
# Time-series regression: regress each asset's returns on factors,
# i.e. one regression per asset. Store the coefficients.

reg_output = {}

for k, g in data_fm.groupby('stock_id'):
    model = ols('R1M_Usd ~ MKT_RF + SMB + HML + RMW + CMA', data=g)
    results = model.fit()

    reg_output[k] = results.params

betas = pd.DataFrame.from_dict(reg_output).T


# %%
# prepping coeficient data to run second round of regressions
loadings = betas.drop('Intercept', axis=1).reset_index(drop=True)
ret = returns.T.reset_index(drop=True)
fm_data = pd.concat([loadings, ret], axis=1)
fm_data = pd.melt(fm_data, id_vars=['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA'])

# running cross section regressions
# Cross-section regression: regress each asset's returns on coefficients obtained in previous step,
# i.e. one regression per time period.
reg_output_2 = {}

for k, g in fm_data.groupby('variable'):
    model = ols('value ~ MKT_RF + SMB + HML + RMW + CMA', data=g)
    results = model.fit()

    reg_output_2[k] = results.params

# refer to the mlfactor book or the fama-macbeth literature for more info on what the gammas stand for
# but you can think of them as an estimate of a given factor's risk premium at a point in time
gammas = pd.DataFrame.from_dict(reg_output_2).T.reset_index().rename({'index': 'date'}, axis=1)


# %%
selected_features = features
y_penalized_train = training_sample['R1M_Usd'].values
X_penalized_train = training_sample[selected_features].values

linear_model = Lasso()
linear_model.fit(X_penalized_train, y_penalized_train)

y_penalized_test = testing_sample['R1M_Usd'].values
X_penalized_test = testing_sample[selected_features].values

y_pred_test = linear_model.predict(X_penalized_test)

mse = mean_squared_error(y_penalized_test, y_pred_test)
r2 = r2_score(y_penalized_test, y_pred_test)
hit_ratio = accuracy_score(np.sign(y_penalized_test), np.sign(y_pred_test))
print(f'MSE: {mse} \nR2: {r2} \nHit Ratio: {hit_ratio}')


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


selected_features = features
y_train = torch.tensor(training_sample['R1M_Usd'].values, device=device, dtype=dtype)
x_train = torch.tensor(training_sample[selected_features].values, device=device, dtype=dtype)
y_test = torch.tensor(testing_sample['R1M_Usd'].values, device=device, dtype=dtype)
x_test = torch.tensor(testing_sample[selected_features].values, device=device, dtype=dtype)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.feed_forward_layers = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        y = self.feed_forward_layers(x)
        return y


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    start_timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        data = data.reshape(batch_size, -1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    end_timer_and_print('Default precision:')