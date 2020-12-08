if __name__ == "__main__":
    
    start_time = time.time()
    #spark_map(1)
    #spark_reduce(3, 4)
    start_spark_session()
    #stocks = get_stocks_from_combination_index(100)
    #calculate_money_today(stocks, [0.2, 0.2, 0.2, 0.2, 0.2])
    
    print("--- %s seconds ---" % (time.time() - start_time))

import random
import pickle
import datetime
from datetime import date, timedelta
import time
import sys
import pandas as pd
import bs4 as bs
import requests
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import pandas_datareader as pdr
import yfinance as yf

from pyspark.sql import SparkSession

number_of_selected_stocks = 5
risk_free_rate = 0.05

# partitions number is recommended equal to computer thread number
partitions = 8
number_of_executed_combinations = 100 * partitions

#we use historical data to approximate mean and variance
start_date='2005-01-01'
end_date ='2016-01-04'

#data location
data_dir = "https://raw.githubusercontent.com/sppro94/bsc/main/stocks"

#we assume to invest money at "start_date" and let's calculate how much we have "today"
invested_money = 100000
today = '2020-01-03'

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.replace(".", "-", 1))

        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)    
        
    return [item.strip() for item in tickers]

sp500_list = save_sp500_tickers()

def read_single_stock(stock):
    data_file_path = "{0}/{1}/{1}.csv".format(data_dir, stock)
    pdf = pd.read_csv(data_file_path).set_index("Date")
    pdf = pdf[['Adj Close']]
    pdf = pdf.rename(columns={"Adj Close": stock})
    pdf = pdf.loc[pdf.index <= end_date]
    pdf = pdf.loc[pdf.index >= start_date]
    return pdf.loc[pdf.index <= end_date]

def read_data(stocks):
    df_list = read_single_stock(stocks[0]).join([read_single_stock(stock) for stock in stocks[1:]],how='outer')
    return df_list

def choose(n, k):
    '''Returns the number of ways to choose k items from n items'''
    reflect = n - k
    if k > reflect:
        if k > n:
            return 0
        k = reflect
    if k == 0:
        return 1
    for nMinusIPlus1, i in zip(range(n - 1, n - k, -1), range(2, k + 1)):
        n = n * nMinusIPlus1 // i
    return n

def iterCombination(index, n, k):
    if index < 0 or index >= choose(n, k):
        return
    n -= 1
    for i in range(k):
        while choose(n, k) > index:
            n -= 1
        yield n
        index -= choose(n, k)
        n -= 1
        k -= 1

def get_random_stock_combination_index():
    n = len(sp500_list)
    r = random.getrandbits(128) % choose(n, number_of_selected_stocks)
    return r

def get_stocks_from_combination_index(r):
    n = len(sp500_list)
    stocks = [sp500_list[stock_index] for stock_index in iterCombination(r, n, number_of_selected_stocks)]
    return stocks

#downloading the data from Yahoo! Finance
def download_data(stocks):
    data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']
    data.columns = stocks
    return data

def download_data_single_day(stocks, day):
    data = web.DataReader(stocks,'yahoo',day,day)['Adj Close']
    data.columns = stocks
    return data
    
#we usually use natural logarithm for normalization purposes
def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns

# OK this is the result of the simulation ... we have to find the optimal portfolio with 
# some optimization technique !!! scipy can optimize functions (minimum/maximum finding)
def statistics(weights, returns):
    portfolio_return=np.sum(returns.mean()*weights)*252
    portfolio_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    return np.array([portfolio_return,portfolio_volatility,portfolio_return/portfolio_volatility])

# [2] means that we want to maximize according to the Sharpe-ratio
# note: maximizing f(x) function is the same as minimizing -f(x) !!!
def	min_func_sharpe(weights,returns):
    return	-statistics(weights,returns)[2] 

#print out mean and covariance of stocks within [start_date, end_date]. There are 252 trading days within a year
def show_statistics(returns):
    print(returns.mean()*252)
    print(returns.cov()*252)

#weights defines what stocks to include (with what portion) in the portfolio
def initialize_weights(stocks):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights

# what are the constraints? The sum of weights = 1 !!!  f(x)=0 this is the function to minimize
def optimize_portfolio(weights,returns, stocks):
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #the sum of weights is 1
    bounds = tuple((0,1) for x in range(len(stocks))) #the weights can be 1 at most: 1 when 100% of money is invested into a single stock
    optimum=optimization.minimize(fun=min_func_sharpe,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraints) 
    return optimum

# optimal portfolio according to weights: 0 means no shares of that given company
def print_optimal_portfolio(optimum, returns):
    print("Optimal weights:", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe ratio:", statistics(optimum['x'].round(3),returns))

def calculate_sharpe_ratio_from_combination_index(index):
    #print("calculating: ", index)
    stocks = get_stocks_from_combination_index(index)
    data = read_data(stocks)
    for stock in stocks:
        if data[stock].isnull().sum() == len(data[stock]):
            #print(index, stocks, 0)
            return 0
    returns = calculate_returns(data)
    weights=initialize_weights(stocks)
    optimum=optimize_portfolio(weights,returns, stocks)
    sta = statistics(optimum['x'].round(3),returns)[2]
    sharpe_ratio= 0 if sta == np.nan else sta
    
    #print(index, stocks, sharpe_ratio)
    return sharpe_ratio
    
def spark_map(r):
    return [r, calculate_sharpe_ratio_from_combination_index(r)]

def spark_reduce(a, b):
    #print("comparing: ", a, b)
    return a if a[1] > b[1] else b

def calculate_money_today(stocks, weights):
    started_stock_values = download_data_single_day(stocks, end_date)
    today_stock_values = download_data_single_day(stocks, today)
    sum_of_money = 0

    for index in range(len(weights)):
        print("price of {} stock in {}: {}".format(stocks[index], end_date, started_stock_values[stocks[index]].values[0]))
        print("price of {} stock in {}: {}".format(stocks[index], today, today_stock_values[stocks[index]].values[0]))
        sum_of_money += invested_money * weights[index] * today_stock_values[stocks[index]].values[0] / started_stock_values[stocks[index]].values[0]

    print("sum of money: {}".format(sum_of_money))
    return sum_of_money

    
def start_spark_session():
    # spark
    spark = SparkSession\
        .builder\
        .appName("Thesis")\
        .getOrCreate()
    c1 = spark.sparkContext\
        .addPyFile("dependencies.zip")
        .parallelize([ get_random_stock_combination_index() for _ in range(1, number_of_executed_combinations + 1)], partitions)\
        .map(spark_map)\
        .reduce(spark_reduce)
    print(c1)

    stocks = get_stocks_from_combination_index(c1[0])
    data = read_data(stocks)
    returns = calculate_returns(data)
    weights=initialize_weights(stocks)
    optimum=optimize_portfolio(weights,returns, stocks)
    calculate_money_today(stocks, weights)

    spark.stop()



    
