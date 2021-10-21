# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:14:15 2021

@author: YS
"""
import sys 
sys.path.append("..") 
from vector import portfolio, data_source
import importlib
from datetime import datetime, timedelta, timezone
import pickle
import os
import json
import tables as tb
import pandas as pd
import talib as ta
from sklearn.decomposition import PCA
import numpy as np
import time
import pymongo
# get original data
importlib.reload(data_source)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


class DataSource(data_source.SourceManager):
    
    # 定义转换数据源命名转换规则: eth -> eth_usdt.spot:binance (MongoDB表名)
    def source_key_map(self, key: str):
        return f"{key}_usdt.spot:binance"
    
    # 定义本地缓存文件命名规则：eth -> eth
    def target_key_map(self, key: str):
        return key


class Signal(data_source.DataManager):
    
    def set_param(self,paramVersion,remark):
        '''
        导入json参数设置
        后续计算指标都将以此处设置的参数进行
        :paramVersion :
        :remark :
        '''
        self.path = '../data'
        with open(self.path+'/'+'setting'+'/'+'sig_setting%s%s.json'%(paramVersion,remark)) as param:
            self.setting = json.load(param)
        return self.setting
    
    def get_basic_data(self):
        df = self.basic_data.copy(deep=True)
        return df


class DataTool():

    def __init__(self,paramVersion,remark):
        '''
        实例化时需要传入参数版本和备注，之后保存数据和读入数据都按照此参数版本和备注进行，无需再另外设置
        '''
        self.symbolsData = pd.DataFrame()
        self.paramVersion = paramVersion #用于统一格式表格计算的参数
        self.path = '../data'
        with open(self.path+'/'+'setting'+'/'+'sig_setting%s%s.json'%(paramVersion,remark)) as param:
            self.setting = json.load(param)
            
        self.MONGODB_HOST = self.setting['MONGODB_HOST']
        self.KLINE_DB = self.setting['KLINE_DB']
        self.symbols = self.setting['symbols']
        self.freqs = self.setting['sigPeriod']
        self.remark = remark
        self.dictDf = {}
        self.begin_time = 0
        self.end_time = 0
        print(f'DataTool.__init__() is called')
        print(f'parameter version:【{self.paramVersion}】')
        print(f'remark:【{self.remark}】')
        print('symbols:',self.symbols)
    def get_data(self, startTime, endTime):
        '''
        取原始数据并合成K线
        :startTime :从数据库取数据的初始时间 数据例为1630928800.0的毫秒格式
        :endTime :从数据库取数据的结束时间
        '''
        ds = DataSource.from_mongodb(
            self.MONGODB_HOST,
            self.KLINE_DB,
            root = '../vector_cache' #存放缓存的默认位置
        )
        # 从数据库拉取一分钟数据
        print('从数据库拉取',self.symbols,'的数据')
        ds.pull(self.symbols, begin=startTime, end=endTime)
        # 合成不同周期k线
        print(self.symbols, self.freqs)
        print('DataTool.get_data() is called')
        ds.resample(self.symbols, self.freqs)
        result = ds.load(self.symbols, self.freqs, startTime, endTime) #result类型是dict
        self.dictDf = result   #函数到这行为止是Channel的代码
        self.symbolsData = result[self.freqs[0]]
        # self.begin_time =  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(startTime))
        # self.end_time =  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(endTime))
        self.begin_time =  time.strftime('%Y%m%d',time.localtime(startTime))
        self.end_time =  time.strftime('%Y%m%d',time.localtime(endTime))
        return self.dictDf
        
    def save_symbols_data(self):
        '''
        保存symbolsData,后缀为时间段，频率，参数版本号，备注
        '''
        filename = 'symbolsData'
        with open(self.path +'/' + 'symbolsData' +'/'+filename +'_'+\
            str(self.begin_time)+str(self.end_time)+'_'+self.freqs[0]+self.paramVersion+self.remark +'.pkl','wb') as f:

            pickle.dump(self.dictDf, f, pickle.HIGHEST_PROTOCOL)
            print(self.path +'/' + 'symbolsData' +'/'+filename +'_'+\
            str(self.begin_time)+str(self.end_time)+'_'+self.freqs[0]+self.paramVersion+self.remark +'.pkl','【save complete!】')
            
    def load_symbols_data(self):
        '''
        读取保存在本地的symbolsData,后缀为时间段，频率，参数版本号，备注
        '''
        filename = 'symbolsData'
        with open(self.path +'/' + 'symbolsData' +'/'+filename +'_'+\
            str(self.begin_time)+str(self.end_time)+'_'+self.freqs[0]+self.paramVersion+self.remark +'.pkl','rb') as f:

            self.dictDf = pickle.load(f)
            self.symbolsData = self.dictDf[self.freqs[0]]
        return self.dictDf    

    def save_signal_data(self,df,freqs,paramVersion,remark):
        '''
        保存计算好的signal数据到本地，后缀为时间段，频率，参数版本号，备注
        '''
        ind = df.index.tolist()
        begin_time = ind[0]
        end_time = ind[-1]
        
        begin_year = str(begin_time)[:10].split('-')[0]
        begin_month = str(begin_time)[:10].split('-')[1]
        begin_day = str(begin_time)[:10].split('-')[2]
        begin_time = begin_year + begin_month + begin_day
        end_year = str(end_time)[:10].split('-')[0]
        end_month = str(end_time)[:10].split('-')[1]
        end_day = str(end_time)[:10].split('-')[2]
        end_time = end_year + end_month + end_day
        filename = 'symbolsSig'
        with open(self.path + '/' + 'symbolsSig' + '/' +filename + '_' +\
            begin_time + end_time +'_' + freqs + paramVersion + remark +'.pkl','wb') as f:

            dic = {freqs:df}
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
        print(self.path + '/' + 'symbolsSig' + '/' +filename + '_' +\
            begin_time + end_time +'_' + freqs + paramVersion + remark +'.pkl','【save complete!】')
        return self.dictDf

    def get_last_ind(self,factor):
        '''
        返回的last_ind是timestamp格式 
        '''
        path = '../data'
        with open(path+'/'+'updateRecord'+'/'+'%s.json'%(factor)) as record:
            record = json.load(record)
        last_ind  =  record['lastUpdateTime']
        print('消息来自get_last_ind 字符串形式的上次最后更新日期: ',last_ind)
        utc = timezone(timedelta())
        year = int(last_ind[:4]) 
#        print('消息来自get_last_ind year',year)
        month = int(last_ind[5:7])
#        print('消息来自get_last_ind month',month)
        day = int(last_ind[8:10])
#        print('消息来自get_last_ind day',day)
        hour = int(last_ind[11:13])
#        print('消息来自get_last_ind day',hour)
        minute = int(last_ind[14:16])
#        print('消息来自get_last_ind day',minute)    
        second = int(last_ind[17:19])
#        print('消息来自get_last_ind day',second) 
        #last_ind = datetime(year, month, day,hour,minute,second, tzinfo=utc).timestamp()
        last_ind = datetime(year, month, day, tzinfo=utc).timestamp()
#        print('消息来自get_last_ind 数字形式的上次最后更新日期: ',last_ind)      
        tupTime = time.localtime(last_ind)
#        print('消息来自get_last_ind tupTime',tupTime)
        return last_ind
        
    def get_update_data(self,factor):
        '''
        获取用于计算更新因子的原始数据
        '''
#        last_ind = self.get_last_ind('absorptionRatio')
        last_ind = self.get_last_ind(factor)
        last_ind -= 3500000 #从上次记录再往前取接近30天的数据
        now_ind = datetime.now().timestamp()
        print('last_ind 最后更新日期往前3500000: ',last_ind)
        print('now_ind 系统当前时间: ',now_ind)
        dic = self.get_data(last_ind,now_ind)
        return dic

    def upload_data(self,df:pd.DataFrame(),factor:str,symbols:list ):
        '''
        把数据传到数据库
        df是以timestamp为索引的dataframe，需要reset_index()
        factor应传入因子名字符串，如absorptionRatio
        '''
        client = pymongo.MongoClient('172.16.11.81', 27017)
        collection = client['multiSymbolsIndicator'][factor]
        df = df.reset_index()
        
        link = ','
        strRes = link.join(symbols)
        symbolList = [strRes]*len(df)
#        symbols = symbols + [np.nan]*(len(df)-len(symbols))
        df['symbols'] = symbolList
        
        print(df.head())
        for index, values in df[:].iterrows():
            bar = values.to_dict()
            values['timestamp'] = str(values['timestamp'])
#            print(bar)
            bar['datetime'] = datetime.strptime(values['timestamp'][:values['timestamp'].find('+')], "%Y-%m-%d %H:%M:%S")
            
            del bar['timestamp']
            collection.create_index([('datetime', pymongo.ASCENDING)], unique=True)
            flt = {'datetime': bar['datetime']}
            collection.update_one(flt, {'$set':bar}, upsert=True)
            print(index,' write complete')  
        
        
        num = str(len(df))
        print(f'Upload {num} {factor} data complete')



class SignalCalculator(Signal):
    
    def cal_sig_roc(self,data_raw:pd.DataFrame,setting:dict): 
        '''
        传入单个symbol的行情数据，参数设置，返回列数为1的DataFrame
        :data_raw :原始行情数据
        :setting :计算指标的参数设置
        '''
        data = data_raw.copy(deep=True)
        roc_res = {}
        roc_param = setting['roc_param']
        for roc_parameter in roc_param[:]:
            tmp = ta.ROC(data['close'],timeperiod = roc_parameter)#新加指标 #***#
            roc_res[ 'roc' + str(roc_parameter)] = tmp
            print(f'roc{str(roc_parameter)} done')
        return pd.DataFrame(roc_res)
    
    def cal_sig_cci(self,data_raw:pd.DataFrame,setting:dict):
        '''
        传入单个symbol的行情数据，参数设置，返回列数为1的DataFrame
        :data_raw :原始行情数据
        :setting :计算指标的参数设置
        '''
        data = data_raw.copy(deep=True)
        cci_res = {}
        cci_param = setting['cci_param']
        for cci_parameter in cci_param[:]:
            tmp = ta.CCI(data['high'],data['low'], data['close'],timeperiod=cci_parameter) #***#
            cci_res['cci' + str(cci_parameter)] = tmp
            print(f'cci{str(cci_parameter)} done')
        return pd.DataFrame(cci_res)
    
    def cal_sig_csi(self,data_raw:pd.DataFrame,setting:dict):
        '''
        传入单个symbol的行情数据，参数设置，返回列数为1的DataFrame
        :data_raw :原始行情数据
        :setting :计算指标的参数设置
        '''
        data = data_raw.copy(deep=True)
        csi_res = {}
        csi_param = setting['csi_param']
        for csi_parameter in csi_param[:]:
            tmp1 = ta.ADXR(data['high'], data['low'], data['close'], timeperiod = csi_parameter) ###参数不一定是14
            tmp2 = ta.ATR(data['high'], data['low'], data['close'], timeperiod = csi_parameter)
            csi_res['csi' + str(csi_parameter)] = tmp1*tmp2
            print(f'csi{str(csi_parameter)} done')
        return pd.DataFrame(csi_res)
    
    
    def ER(self,lst:list):
        '''
        :lst :数据列表
        '''
        arr = np.array(lst)
        a = arr[1:]
        b = arr[:-1]
        lst_diff = abs(a - b)
        c = lst_diff.sum()
        d = abs(arr[-1] - arr[0])
        er = d/c
        return er
    
    def cal_sig_er(self,data_raw:pd.DataFrame,setting:dict):
        data = data_raw.copy(deep=True)
        er_res = {}
        er_param = setting['er_param']
        for er_parameter in er_param[:]:
            tmp = data['close'].rolling(er_parameter).apply(lambda x: self.ER(x))  #***#
            er_res['er' + str(er_parameter)] = tmp
            print(f'er{str(er_parameter)} done')
        return pd.DataFrame(er_res)
    
    def handle_symbol(self, symbol: str, freq: str, data: pd.DataFrame) -> pd.DataFrame:
        if freq == "5min": ##################################################################
            setting = self.setting
            
            if setting['indicator_name'] == 'roc':
                res = self.cal_sig_roc(data,setting)
                print(f'Calculate 【{symbol} roc】 complete')
                
            if setting['indicator_name'] == 'cci':
                res = self.cal_sig_cci(data,setting)
                print(f'Calculate 【{symbol} cci】 complete')
            
            if setting['indicator_name'] == 'csi':
                res = self.cal_sig_csi(data,setting)
                print(f'Calculate 【{symbol} csi】 complete')            
                
            if setting['indicator_name'] == 'er':
                res = self.cal_sig_er(data,setting)
                print(f'Calculate 【{symbol} er】 complete')
            return  res 
        else:
            return super().handle_symbol(symbol, freq, data) 
        
    def cal_avg_roc(self):
        df = self.basic_data.copy(deep=True)
        setting =self.setting
        roc_param = setting['roc_param']
        for roc_parameter in roc_param[:]:
            df_roc = df.loc[:, pd.IndexSlice[:, "roc"+str(roc_parameter)]] ###取二级列索引
            df['avg_roc'+str(roc_parameter)] = df_roc.mean(axis=1)
            print(f'avg_roc{str(roc_parameter)} done')
        return df.iloc[:,-len(roc_param):].dropna(how='all',axis=0) 
    
    def cal_avg_cci(self):
        df = self.basic_data.copy(deep=True)
        setting = self.setting
        cci_param = setting['cci_param']
        for cci_parameter in cci_param[:]:
            df_cci = df.loc[:, pd.IndexSlice[:, "cci"+str(cci_parameter)]] ###取二级列索引
            df['avg_cci'+str(cci_parameter)] = df_cci.mean(axis=1)
            print(f'avg_cci{str(cci_parameter)} done')
        return df.iloc[:,-len(cci_param):].dropna(how='all',axis=0)
    
    def cal_avg_csi(self):
        df = self.basic_data.copy(deep=True)
        setting = self.setting
        csi_param = setting['csi_param']
        for csi_parameter in csi_param[:]:
            df_csi = df.loc[:, pd.IndexSlice[:, "csi"+str(csi_parameter)]] ###取二级列索引
            df['avg_csi'+str(csi_parameter)] = df_csi.mean(axis=1)
            print(f'avg_csi{str(csi_parameter)} done')
        return df.iloc[:,-len(csi_param):].dropna(how='all',axis=0)
    
    def cal_avg_er(self):
        df = self.basic_data.copy(deep=True)
        setting = self.setting
        er_param = setting['er_param']
        for er_parameter in er_param[:]:
            df_er = df.loc[:, pd.IndexSlice[:, "er"+str(er_parameter)]] ###取二级列索引
            df['avg_er'+str(er_parameter)] = df_er.mean(axis=1)
            print(f'avg_er{str(er_parameter)} done')
        return df.iloc[:,-len(er_param):].dropna(how='all',axis=0)
    
    def cal_pca(self,pctArray):
#        pca = PCA(n_components=5)
        num = len(self.setting['symbols'])
        pca = PCA(n_components=num)
        pca.fit(pctArray)
        res = pca.explained_variance_ratio_[0]
        return pca.explained_variance_ratio_[0]

    
    def cal_ar(self,data_raw,ar_param):
        symbolsPct = data_raw.loc[:, pd.IndexSlice[:, "close"]].pct_change().dropna()
        pcaList = []
        for i in range(len(symbolsPct)-ar_param+1):
            pctArray = np.array(symbolsPct.iloc[i:i+ar_param])
            pcaList.append(self.cal_pca(pctArray))
        
        symbolsPca = symbolsPct.iloc[ar_param-1:]
        symbolsPca['absorption'] = pcaList
        data_raw["ar"+str(ar_param)] = symbolsPca['absorption']
        colname = data_raw.columns.tolist()[-1]
        print(f'{colname[0]} done')
#        self.basic_data = data_raw
        return data_raw
    
    def cal_total_ar(self):
        data = self.basic_data.copy(deep=True)
        setting = self.setting
        ar_param = setting['ar_param']
        for ar_parameter in ar_param:
            data = self.cal_ar(data,ar_parameter)
        res = data.iloc[:,-len(ar_param):].dropna(how='all',axis=0)
        return res
    
class Updater(DataTool,SignalCalculator):
    
    def __init__(self,DataToolparamVersion = '_v1',DataToolremark = '_origin2',dictDf:dict=None,instanceId=0):
#    def __init__(self,paramVersion,remark ,dictDf:dict=None,instanceId=0):
        '''
        :DataToolparamVersion :父类DataTool的初始化参数，用于确定DataTool获取行情信息的json格式参数
        :DataToolremark :同paramVersion
        :dictDf :父类SignalCalculator的初始化参数，用于计算指标，默认为空
        :instanceId :实例的ID，用于区分不同updater对象
        '''
        DataTool.__init__(self,DataToolparamVersion,DataToolremark) #调用父类构造函数
        SignalCalculator.__init__(self,dictDf) #调用父类构造函数
        self.id = instanceId
        
    def get_new(self,factor:str,SignalCalculatorparamVersion:str,SignalCalculatoremark:str,save=False,upload=False):
        '''
        :factor :因子名
        :paramVersion :要计算对应因子的参数版本
        :remark :要计算对应因子的参数备注 例:sig_setting_v1_er.json
        :upload :默认为False，为True时将计算好的因子数据上传到数据库
        '''
        update_setting = self.set_param(SignalCalculatorparamVersion,SignalCalculatoremark) #set_param是Signal的方法，设置参数版本和备注，get_update_data()会用到
        print('*****************开始获取行情数据*****************')
        symbolsData = self.get_update_data(factor) #传入factor名称会自动获取计算对应指标所需行情数据,
        print('symbolsData get!')
        print('*****************行情数据获取完毕*****************')
        if save == True:
            self.save_symbols_data() #保存行情数据到本地
            print('*****************行情数据保存完毕*****************')
        
        calculator = SignalCalculator(symbolsData) #实例化SignalCalculator，传入的参数是字典形式{'5min':df}
        calculator.set_param(SignalCalculatorparamVersion,SignalCalculatoremark) #传入的参数版本和指标备注会传给SignalCalculator实例，自动调用对应指标计算函数
        print('*****************开始计算因子数据*****************')
        if factor == 'efficiencyRatio':
            calculator.prepare_data()
            result = calculator.cal_avg_er()
            
        if factor == 'roc':
            calculator.prepare_data()
            result = calculator.cal_avg_roc()
            
        if factor == 'cci':
            calculator.prepare_data()
            result = calculator.cal_avg_cci()        
        
        if factor == 'csi':
            calculator.prepare_data()
            result = calculator.cal_avg_csi()
        
        if factor == 'absorptionRatio':
            #ar的计算方式和其他指标不一样,调用的接口不是cal_avg_xxx() 
            result = calculator.cal_total_ar()
        
        last_ind = self.get_last_ind(factor)
#        print('调用get_last_ind获取的last_ind',last_ind)
        tupTime = time.localtime(last_ind)
#        print('tupTime',tupTime)
        stadardTime = time.strftime("%Y-%m-%d %H:%M:%S", tupTime)
#        print('毫秒格式上次更新数据最后日期',last_ind)
#        print('上次更新数据最后日期',stadardTime)
        result = result.loc[stadardTime:]
        print('*****************因子数据计算完毕*****************')
        if save == True:
            self.save_signal_data(result,'5mn',SignalCalculatorparamVersion,SignalCalculatoremark)
            print('*****************因子数据保存完毕*****************')
        if upload == True:
            print('*****************开始上传因子到数据库*****************')
            uploadData = result.copy(deep=True)
            uploadData.columns = [tu[0] for tu in result.columns.tolist()]
            tableName = factor + 'EUL'
            self.upload_data(uploadData,tableName,update_setting['symbols'])
            print('*****************因子数据上传完毕*****************')
        return result