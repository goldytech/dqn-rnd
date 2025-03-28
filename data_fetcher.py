# data_fetcher.py
import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance API for multiple tickers.
    
    :param tickers: List of stock ticker symbols
    :param start_date: Start date for fetching data
    :param end_date: End date for fetching data
    :return: Dictionary of DataFrames with stock data for each ticker
    """
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start=start_date, end=end_date)
    return data

def fetch_technical_data(data):
    """
    Calculate technical indicators from stock data.
    
    :param data: DataFrame with stock data
    :return: DataFrame with technical indicators
    """
    data['SMA'] = data['Close'].rolling(window=20).mean()  # Simple Moving Average
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()  # Exponential Moving Average
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).mean()))
    return data

def fetch_fundamental_data(ticker):
    """
    Fetch fundamental data from Yahoo Finance API.
    
    :param ticker: Stock ticker symbol
    :return: DataFrame with fundamental data
    """
    stock = yf.Ticker(ticker)
    fundamentals = stock.financials.T
    return fundamentals

def fetch_fundamental_data(tickers):
    """
    Fetch fundamental data from Yahoo Finance API for multiple tickers.
    
    :param tickers: List of stock ticker symbols
    :return: Dictionary of DataFrames with fundamental data for each ticker
    """
    fundamentals = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        fundamentals[ticker] = stock.financials.T
    return fundamentals

