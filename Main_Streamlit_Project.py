import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as mp
import plotly.express as px
import seaborn as sns
import requests
from scipy import stats
from PIL import Image
from datetime import date

option = st.sidebar.selectbox("What do you want to do?",('Stock Analysis','Correlation Analysis'))
#my_image = Image.open('finances.jpg')
#image = my_image.resize((400,200))
#st.image(my_image)

df_tickers = pd.read_csv('tickers.csv')
if option == 'Stock Analysis':
    symbol = st.sidebar.selectbox("Ticker", df_tickers.iloc[:,0].values)
    start = st.sidebar.date_input('Start date', date.today()+pd.DateOffset(years=-1))
    end = st.sidebar.date_input('End date')

    stock = yf.download(symbol, start = start, end = end)
    logret = np.log(stock.Close).diff().dropna()
    st.write(""" # Quantitative Finance """)
    st.write(""" ## Resume """)
    st.write(""" Ticker: """, symbol)
    st.write(' Last close: ' + str(round(stock['Close'][-1],2)))
    #st.dataframe(stock)
    #st.download_button('Press to Download', stock.to_csv().encode('utf-8'), file_name = symbol+start.strftime("%m%d%Y")+".csv", mime = "text/csv", key = 'download-csv')
   
     #%% Simulation
    # Given parameters
    RiskFree = 0.0429
    T = 1
    n = len(logret)
    dt = T / n
    Current_Price = stock['Close'][-1]

    # Calculate standard deviation and volatility
    StDev = (np.std(logret) * np.sqrt(n)) * np.sqrt(T)
    Vol = StDev * np.sqrt(T)

    # Calculate mean return
    Mean = (RiskFree - (StDev**2) / 2) * T

    # Simulate stock prices using Monte Carlo simulation
    M = 100000
    z = np.random.randn(M, 1)
    Sim = Mean + Vol * z
    Sim_logret= Current_Price * np.exp(Sim)

    # Calculate average simulated price
    Pr_Sim = np.mean(Sim_logret)
    Simulated_Price = np.mean(Pr_Sim)

    # Calculate arbitrage percentage
    Arbitrage = np.log(Simulated_Price / Current_Price) * 100
    st.write("Current Price:", Current_Price)
    st.write("Simulated Price:", Simulated_Price)
    st.write("Arbitrage %:", Arbitrage)
    

    st.write(""" ## Close Prices """)
    fig = px.line(stock, y='Close')
    st.plotly_chart(fig, use_conatainer_width=True)

    st.write(""" ## Returns """)
    st.write(""" Descriptive statistics of log-returns """)
    st.dataframe(logret.describe())
 
    #%% Functions
    def normal(mean, std, color="black"):
        x = np.linspace(mean-6*std, mean+6*std, 200)
        p = stats.norm.pdf(x, mean, std)
        mp.plot(x, p, color, linewidth=1)

    hist_ret = mp.figure(figsize=(8,6))
    mp.title('Returns Histogram')
    sns.histplot(logret, stat='density', color='blue')
    normal(logret.mean(), logret.std())
    st.pyplot(hist_ret)

    #%% Simulated Returns
    #st.write(""" ## Simulated Returns """)
    #st.write(""" Descriptive statistics of log-returns """)
    #st.dataframe(Sim_logret.describe())



if option == 'Correlation Analysis':
     portfolio = st.sidebar.multiselect('Tickers', df_tickers.iloc[:,0].values)
     start = st.sidebar.date_input('Start date', date.today()+pd.DateOffset(years=-1))
     end = st.sidebar.date_input('End date')
     
     st.write(""" ## Your Portfolio """)
     st.markdown(portfolio)

     df_port = yf.download(portfolio, start=start, end=end)['Adj Close']
     st.write(""" ## Close prices """)
     st.dataframe(df_port)

     ret = np.log(df_port).diff().dropna()
     cov_mat = ret.cov()
     st.write(""" ## Covariance Matrix """)
     cov_mat

     st.write(""" ## Correlation Matrix """)
     fig = mp.figure()
     sns.heatmap(ret.corr(), annot=True, cmap='Reds', center=1, linewidths=.5)
     st.pyplot(fig)

