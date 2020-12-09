import random
import pickle
import datetime
from datetime import date, timedelta
import time
import sys
import getopt
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

# partitions number is recommended equal to computer's threads number
partitions = 8
units = 100
number_of_executed_combinations = units * partitions
run_spark = True

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

def download_data_single_day(stocks, day):
    data = web.DataReader(stocks,'yahoo',day,day)['Adj Close']
    data.columns = stocks
    return data
    
def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns

# stats of a portfolio: return, volatility and Sharpe-ratio
def stats(weights, returns):
    p_return=np.sum(returns.mean()*weights)*252
    p_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    return np.array([p_return,p_volatility,p_return/p_volatility])

# Minimize -f(x) to get maximum f(x)
def	min_sharpe(weights,returns):
    return	-stats(weights,returns)[2] 

def initialize_weights(stocks):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights

def optimize_portfolio(weights,returns, stocks):
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #the sum of weights is 1
    bounds = tuple((0,1) for x in range(len(stocks))) #the weights can be 1 at most: 1 when 100% of money is invested into a single stock
    optimum=optimization.minimize(fun=min_sharpe,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraints) 
    return optimum

def slope_of_CAL(p_return, p_volatility):
    return np.arctan((p_return-risk_free_rate)/p_volatility)

# calculate the slope of CAL from combination index(r)
def calculate_slope(index):
    stocks = get_stocks_from_combination_index(index)
    data = read_data(stocks)
    for stock in stocks:
        if data[stock].isnull().sum() == len(data[stock]):

            return 0
    returns = calculate_returns(data)
    weights=initialize_weights(stocks)
    optimum=optimize_portfolio(weights,returns, stocks)
    s = stats(optimum['x'].round(3),returns)

    slope = 0 if s[2] == np.nan else slope_of_CAL(s[0], s[1])
    return slope
    
def spark_map(r):
    return [r, calculate_slope(r)]

def spark_reduce(a, b):
    return a if a[1] > b[1] else b

def show_statistics(stocks, returns, optimum):
    started_stock_values = download_data_single_day(stocks, end_date)
    today_stock_values = download_data_single_day(stocks, today)
    sum_of_money = 0
    weights = optimum['x'].round(3)

    print("Chosen portfolio: ", stocks)
    print("Optimal weights:", weights)
    print("Expected return, volatility and Sharpe ratio:", stats(weights, returns))

    print("\nIn {}, we invest money on: ".format(end_date))
    for index in range(len(weights)):
        print("{} stock: {} USD".format(stocks[index], weights[index].round(3) * invested_money))

    print("\nStocks' prices: ")

    for index in range(len(weights)):
        print("Price of {} stock in {}: {}".format(stocks[index], end_date, started_stock_values[stocks[index]].values[0].round(3)))
        print("Price of {} stock in {}: {}".format(stocks[index], today, today_stock_values[stocks[index]].values[0].round(3)))
        sum_of_money += invested_money * weights[index] * today_stock_values[stocks[index]].values[0].round(3) / started_stock_values[stocks[index]].values[0].round(3)

    print("\nIn {}, we have totally: {}".format(today, sum_of_money))

    
def start_spark_session():
    # spark
    spark = SparkSession\
        .builder\
        .appName("Thesis")\
        .getOrCreate()
    c1 = spark.sparkContext\
        .parallelize([ get_random_stock_combination_index() for _ in range(1, number_of_executed_combinations + 1)], partitions)\
        .map(spark_map)\
        .reduce(spark_reduce)
    print(c1)

    stocks = get_stocks_from_combination_index(c1[0])
    data = read_data(stocks)
    returns = calculate_returns(data)
    weights=initialize_weights(stocks)
    optimum=optimize_portfolio(weights,returns, stocks)
    show_statistics(stocks, returns, optimum)

    spark.stop()

def show_efficient_frontier():
    r = []
    v = []
    max_slope = 0
    choosen_r = 0
    choosen_v = 0
    for _ in range(1, number_of_executed_combinations + 1):
        index = get_random_stock_combination_index()
        print("Processing index:", index)
        stocks = get_stocks_from_combination_index(index)
        data = read_data(stocks)

        skip_combination = False
        for stock in stocks:
            if data[stock].isnull().sum() == len(data[stock]):
                skip_combination = True
        if (skip_combination):
            continue

        returns = calculate_returns(data)
        weights=initialize_weights(stocks)
        optimum=optimize_portfolio(weights,returns, stocks)
        s=stats(optimum['x'].round(3),returns)
        
        if (s[0] <= risk_free_rate):
            continue

        slope = slope_of_CAL(s[0], s[1])

        #chose the set of stocks with highest value of CAL's slope
        if (max_slope < slope):
            max_slope = slope
            choosen_r = s[0]
            choosen_v = s[1]
            p_stocks = stocks
            p_returns = returns
            p_optimum = optimum

        r.append(s[0])  # set of return
        v.append(s[1])  # set of volatility
    
    show_statistics(p_stocks, p_returns, p_optimum)

    plt.axline((0, risk_free_rate), (choosen_v, choosen_r))
    ax = plt.gca()
    ax.set_ylim([0,0.5])
    ax.set_xlim([0,1])
    plt.xlabel('Volatility')
    plt.ylabel('Return(%)')
    plt.plot(v, r, "ro")
    plt.show()

def read_argvs(argv):
    help_text = "help: efficient_frontier.py [-p <partitions_number>] [-u <calculated_units_number>] [-e] [-s <location_to_stock_data>]"
    try:
        opts, args = getopt.getopt(argv,"hp:u:es:")
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(help_text)
            sys.exit()
        elif opt == "-p":
            globals()["partitions"] = int(arg)
        elif opt =="-u":
            globals()["units"] = int(arg)
        elif opt == "-e":
            globals()["run_spark"] = False
        elif opt == "-s":
            globals()["data_dir"] = arg

if __name__ == "__main__":
    read_argvs(sys.argv[1:])
    number_of_executed_combinations = units * partitions
    print("Partitions: ", partitions)
    print("Units per partitions: ", units)
    print("Number of executed stock sets: ", number_of_executed_combinations)
    print("Data location: ", data_dir)

    start_time = time.time()
    
    if run_spark:
        start_spark_session()
    else:
        show_efficient_frontier()
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
