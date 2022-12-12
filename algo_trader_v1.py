#for getting data and performing trades
from itertools import count
from os import stat
import alpaca_trade_api as tradeAPI
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import datetime
import math

#for applying indicators
import pandas as pd
import pandas_ta as ta

#writting data to excel file
from openpyxl import Workbook

#for plotting
import mplfinance
from mpl_finance import candlestick_ohlc
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
from plotly import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

#getting API keys
API_KEY = ""
API_SECRET = ""
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'

#------------------------------------------------------------------------------------------------------------------------------------------#
class TradingBot(object):
    def __init__(self):
        #initalizing all needed variables
        self.key_id = API_KEY
        self.secret_key = API_SECRET
        self.base_url = 'https://paper-api.alpaca.markets'
        self.data_url = 'https://data.alpaca.markets'
        self.alpaca = tradeAPI.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')
        self.symbol = 'AAPL'
        self.equity = float(10000) #change to alpaca equity for live
        print("Starting Capital:",self.equity)

        
#------------------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------------------------------#
    def startTesting(self):
        print("\nBacktesting Starting...")

        startDate = "2022-03-01"
        self.endDate = endDate = "2022-03-10"
        #pulling historic data from Alpaca API into a dataframe
        #need 30 min timeframe to line up perfectly with real time data
        self.cleanDF = self.tickerDF = self.alpaca.get_bars(self.symbol, TimeFrame(30, TimeFrameUnit.Minute), startDate , endDate, adjustment='raw').df
        
        self.cleanDF['Date'] = self.cleanDF.index
        self.cleanDF['Date'] = self.cleanDF['Date'].dt.tz_convert(tz='US/Mountain')
        self.cleanDF = self.cleanDF.set_index('Date')
        self.cleanDF =self.cleanDF.between_time('7:30', '13:30')
        self.cleanDF['Date'] = self.cleanDF.index

        
        # get only market hours
        self.tickerDF['timestamp'] = self.tickerDF.index
        self.tickerDF['timestamp']= self.tickerDF['timestamp'].dt.tz_convert(tz='US/Mountain')
        self.tickerDF = self.tickerDF.set_index('timestamp')
        self.tickerDF = self.tickerDF.between_time('7:30', '13:30')
        print(self.tickerDF)
        #applying and appending TA's to tickerDF
        self.tickerDF.ta.macd(close ='close', fast=12, slow=26, signal=9, append=True)
        #applying and appending EMA to tickerDF
        self.tickerDF.ta.ema(close ='close', length = 40, append=True)
        #applying and appending 8 ATR to tickerDF
        self.tickerDF.ta.atr(high = 'high', low = 'low',close = 'close', length = 8, append=True)
        #ATRr_8

        pd.set_option("display.max_columns", None)
        
        #print(self.tickerDF) #MACD is macd line, MACDh is macd histogram, MACDs is macd signal line

        self.processEachCandle()
#------------------------------------------------------------------------------------------------------------------------------------------#
    def processEachCandle(self):
        print("----Beginning Processing Candles----")
        df = self.tickerDF

        startingBal = self.equity
        endingBal = self.equity
        shares = int(0)
        wincount = int(0)
        losscount = int(0)
        avgWin = 0
        avgLoss = 0
        tradecount = int(0)
        buyPrice = float(0)
        s_r_time = []



        #setting up support and resistance lists and functions
        #using 5 candle fractal to determine if support or resistance
        #https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9
        s_r_Levels = []
     
        def isSupport(df, i):
            support = df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
            return support

        def isResistance(df, i):
            resistance = df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['low'][i-2]
            return resistance

        def isFarFromLevel(l):
            s = np.mean(df['high']-df['low'])
            return np.sum([abs(l-x) < s for x in s_r_Levels]) == 0

        def inBuyRange(currentPrice, currentLevel, ATR):
            if (currentPrice >= currentLevel and currentPrice <= currentLevel+ATR):
                return True
            else:
                return False
        
        def inSellRange(currentPrice, currentLevel, ATR):
            if (currentPrice <= currentLevel and currentPrice >= currentLevel-ATR):
                return True
            else:
                return False

        #gathering support and resistances via 5 candle fractal
        for counter in range(len(df)):

            #testing if current data is at support or resistance
            if (counter-2 >= 0):
                if isSupport(df, counter-2):
                    l = df['low'][counter-2]

                    if isFarFromLevel(l):
                        s_r_Levels.append((counter-2,df['low'][counter-2]))
                        s_r_time.append(df.index[counter-2])
                        print("Support level at",df['low'][counter-2],"at time",df.index[counter-2])

                elif isResistance(df, counter-2):
                    l = df['high'][counter-2]

                    if isFarFromLevel(l):
                        s_r_Levels.append((counter-2,df['high'][counter-2]))
                        s_r_time.append(df.index[counter-2])
                        print("Resistance level at",df['high'][counter-2],"at time",df.index[counter-2])

        #converting s_r_Levels to double since they are currently tuple
        doubleLevels = np.array(s_r_Levels, dtype=np.float32)

        #implementing strategy
        for counter in range(len(df)):

            #risking 10% of cap of position
            avaliableCap = 0.10*endingBal

            tempclose = df['close'][counter]
            tempATR = df['ATRr_8'][counter]
            tempTime = df.index[counter]
            timeToBuy = False
            timeToSell = False
            numShares = np.floor(avaliableCap / tempclose)

            #buying if within ATR of support
            if(avaliableCap > tempclose and shares == 0): #currently only allowing 1 position open at a time for damage control
                for i in range(len(doubleLevels)):

                    templevel = doubleLevels[i][1]
                    if(inBuyRange(tempclose,templevel,tempATR)):
                        timeToBuy = True

                    if (timeToBuy == True):
                        print("\nBuying",numShares,"shares at",tempclose,"at time",tempTime)
                        buyPrice = tempclose
                        shares += numShares
                        endingBal -= numShares * tempclose
                        print("Current Balance:",endingBal,"\nShares held:",shares)
                        break

            #selling if within ATR of resistance
            if (shares > 0):
                for i in range(len(doubleLevels)):

                    templevel = doubleLevels[i][1]
                    if(inSellRange(tempclose,templevel,tempATR)):
                        timeToSell = True
                    
                    if(timeToSell == True):
                        print("\nSelling",shares,"shares at",tempclose,"at time",tempTime)
                        endingBal += shares * tempclose
                        shares = 0
                        tradecount += 1
                        print("Current Balance:",endingBal)

                        if(tempclose > buyPrice):
                            print("****Win****")
                            wincount += 1
                        else:
                            print("****Loss****")
                            losscount += 1
                        break
                        
        #if there's any shares left at end of testing, sell them all based on final close price of df
        if (shares != 0):
            print("\nClosing all positions")
            tradecount += 1
            price = float(df.tail(1)['close'])
            endingBal = endingBal + (shares * price)
            print("Selling",shares,"shares at",float(df.tail(1)['close']))
            shares = 0

            if(price > buyPrice):
                        print("****Win****")
                        wincount += 1
            else:
                        print("****Loss****")
                        losscount += 1

        #printing results of session
        print("\n----Starting Balance:", startingBal)
        print("----Ending Balacne:", endingBal)
        print("----Number of Trades:",tradecount)
        print("----Number of Wins:",wincount)
        print("----Average Win:",avgWin/wincount)
        print("----Number of Losses",losscount)
        print("----Average Loss:",avgLoss/losscount)
        print("----Win Rate:",(wincount/(wincount+losscount))*100)
        if(endingBal > startingBal):
            print("\nNet gain of",endingBal-startingBal)
        if(startingBal > endingBal):
            print("\nNet loss of",startingBal-endingBal)


        def printResults():
            #initalizing chart

            df = self.tickerDF
            fig = make_subplots(rows=2, cols=2)
            fig.layout = dict(xaxis=dict(type="category"))

            # Candlestick chart for pricing
            fig.append_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'], #plotting candlesticks
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    showlegend=False
                ), row=1, col=1
            )

            for i in range(len(doubleLevels)):
                fig.add_shape(type="line", x0=s_r_time[i],
                                            y0=doubleLevels[i][1],
                                            x1=max(s_r_time),
                                            y1=doubleLevels[i][1])
                
            """
            plt.hlines(level[1],xmin=df['Date'][level[0]],\
                        xmax=max(df['Date']),colors='blue')
            
            # Fast Signal (%k)
            fig.append_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_12_26_9'], #plotting macd line
                    line=dict(color='#ff9900', width=2),
                    name='macd',
                    # showlegend=False,
                    legendgroup='2',
                ), row=2, col=1
            )
            # Slow signal (%d)
            fig.append_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACDs_12_26_9'], #plotting signal line
                    line=dict(color='#000000', width=2),
                    # showlegend=False,
                    legendgroup='2',
                    name='signal'
                ), row=2, col=1
            )
            # Colorize the histogram values
            colors = np.where(df['MACDh_12_26_9'] < 0, '#000', '#ff9900') #colorizing when macd histogram is below 0
            # Plot the histogram
            fig.append_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACDh_12_26_9'],
                    name='histogram',
                    marker_color=colors,
                ), row=2, col=1
            )


            #price line
            fig.append_trace(
                go.Scatter(
                    x = df.index,
                    y = df['open'],
                    line=dict(color='#ff9900', width=1),
                    name='open',
                    # showlegend=False,
                    legendgroup='1',
                ),row=1, col=1
            )
            """
            
                        
            # Make it pretty
            layout = go.Layout(
                plot_bgcolor='#efefef',
                # Font Families
                font_family='Monospace',
                font_color='#000000',
                font_size=20,
                xaxis=dict(
                    rangeslider=dict(
                        visible=False
                    )
                )
            )
            # Update options and show plot
            fig.update_layout(layout)
            fig.show()
        
        printResults()

#------------------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    trader = TradingBot()
    trader.startTesting()
#------------------------------------------------------------------------------------------------------------------------------------------#


"""
            plt.rcParams['figure.figsize'] = [12,12]
            plt.rc('font', size=14)

            fig, ax = plt.subplots()

            candlestick_ohlc(ax,self.cleanDF.values,width=0.009, \
                            colorup='green', colordown='red', alpha=0.8)


            plt.gca().xaxis.set_major_locator(dates.HourLocator())
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m %d %Y %H:%M'))

            plt.title('Trading Results')
            plt.xticks(rotation = 30)
            plt.grid()
            plt.gcf().autofmt_xdate()
            fig.tight_layout()

            for level in s_r_Levels:
                plt.hlines(level[1],xmin=df['Date'][level[0]],\
                        xmax=max(df['Date']),colors='blue')
"""

"""
            plt.rcParams['figure.figsize'] = [12,12]
            plt.rc('font', size=14)
            plt.figure()
            df2 = self.cleanDF
            width = 0.009
            width2 = 0.001

            up = df2[df2['close']>=df2['open']]
            down = df2[df2['close']<df2['open']]
            
            col1 = 'green'
            col2 = 'red'

            #plot up prices
            plt.bar(up.index,up.close-up.open,width,bottom=up.open,color=col1)
            plt.bar(up.index,up.high-up.close,width2,bottom=up.close,color=col1)
            plt.bar(up.index,up.low-up.open,width2,bottom=up.open,color=col1)

            #plot down prices
            plt.bar(down.index,down.close-down.open,width,bottom=down.open,color=col2)
            plt.bar(down.index,down.high-down.open,width2,bottom=down.open,color=col2)
            plt.bar(down.index,down.low-down.close,width2,bottom=down.close,color=col2)

            #plot s/r lines
            for level in s_r_Levels:
                plt.hlines(level[1],xmin=df2['Date'][level[0]],\
                        xmax=max(df2['Date']),colors='blue')
        
            plt.gca().xaxis.set_major_locator(dates.HourLocator())

            plt.grid()
            #rotate x-axis tick labels
            plt.xticks(rotation=30, ha='right')
            plt.show()

        printResults()
"""