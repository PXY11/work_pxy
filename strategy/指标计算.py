# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:23:21 2021

@author: Mr.P
"""
import os
import sys
import talib as ta
syspath = os.path.split(os.path.abspath("."))[0]
sys.path.append(syspath)
import univ3api.simulation as sim
import univ3api.utils as utils
import importlib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from dateutil.parser import parse
import numpy as np
dataCoin = pd.read_csv('../data/dataCoin.csv')
swapdata = pd.read_csv('../data/swapdata.csv',index_col=0)
pricedata = pd.read_csv('../data/pricedata.csv',index_col=0)
priceDt =  pd.read_csv('../data/priceDt.csv',index_col=0)
priceDay = pd.read_csv('../data/priceDay.csv',index_col=0)
#计算指标
dataCoin['pctChange'] = dataCoin['close'].pct_change()
dataCoin['pctStd'] = ta.STDDEV(dataCoin['pctChange'], 24*5)*(24**0.5)
dataCoin['pctStdVma'] = ta.SUM(dataCoin['pctStd']*dataCoin['volume'], 24*50) / ta.SUM(dataCoin['volume'], 24*50)
dataCoin['closeVma'] = ta.SUM(dataCoin['close']*dataCoin['volume'], 24*50) / ta.SUM(dataCoin['volume'], 24*50)
dataCoin['pctStdVSma'] = ta.EMA(dataCoin.pctStdVma, 24*24)
dataCoin['pctStdVLma'] = ta.MA(dataCoin.pctStdVma, 24*140)
dataCoin['SmaLowerLma'] = dataCoin['pctStdVSma']<dataCoin['pctStdVLma']
dataCoinStat = dataCoin.describe()['pctStd']
dataCoin['VolLowerQuantile25'] = dataCoin['pctStd'] < 0.032949
#%%
dataCoin['datetime'] = pd.to_datetime(dataCoin['datetime'])
dataCoin_test_period = dataCoin[dataCoin['datetime']>parse('2021-05-10 00:00:00')]
df_tmp = dataCoin_test_period[['datetime','SmaLowerLma','VolLowerQuantile25']]
df_tmp['timestamp'] = [0]*len(df_tmp)
df_tmp['sqrtPriceX96'] = [0]*len(df_tmp)
df_tmp['price'] = [0]*len(df_tmp)
df_tmp = df_tmp[['timestamp','sqrtPriceX96','price','datetime','SmaLowerLma','VolLowerQuantile25']]
df_tmp['ind'] = [i for i in range(len(pricedata),len(pricedata)+len(df_tmp))]
df_tmp.set_index('ind',inplace=True)
pricedata['SmaLowerLma'] = [np.nan]*len(pricedata)
pricedata['VolLowerQuantile25'] = [np.nan]*len(pricedata)
pricedata['datetime'] = pd.to_datetime(pricedata['datetime'] )
#%%
pricedata_tmp = pricedata.append(df_tmp)
pricedata_tmp = pricedata_tmp.sort_values(by='datetime')
pricedata_tmp = pricedata_tmp.fillna(method = 'bfill')
pricedata_res = pricedata_tmp.loc[pricedata.index]
pricedata_res.set_index('datetime',inplace=True)
pricedata_res['trend'] = priceDt['trend']
#%%
pricedata_res.to_csv('../data/pricedata_res.csv',index=True)








