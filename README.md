# Algorithmic-Stock-Trading-in-Python
This program is still an extreme WIP. The main premise of it is to pull historical data from the Alpaca API, store it into a pandas data frame, find the major support and resistances via fractal trend analysis, then implement select strategies and evaluate their validity. 

It does this by keeping a list of all buys/sells and calculates an overall account balance based on the recorded orders. This is all done in Alpaca’s virtual market, so no real money is being used.

Ideally, a strategy could be tested with this program, then implemented in a similar program in a live market environment. 

