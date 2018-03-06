import urllib
#import urllib2
import urllib.request
import urllib.parse
import urllib.error
import sys
import json
import os
import datetime
import time

class GetData:
    #------------------------------------------------
    def GetStockDataByCode(self, code, startdate, enddate):
        host = 'http://stock.market.alicloudapi.com'
        path = '/sz-sh-stock-history'
        method = 'GET'
        appcode = '04c939dd89934a53883a4a4397c7358d'
        # exp:              'begin=2015-09-01&code=600004&end=2015-09-02'
        querys = str.format('begin={}&code={}&end={}', startdate, code, enddate)
        bodys = {}
        url = host + path + '?' + querys

        request = urllib.request.Request(url)
        
        request.add_header('Authorization', 'APPCODE ' + appcode)
        try:
            response = urllib.request.urlopen(request)
        except:
            print('request err wait 60s\r\n')
            time.sleep(60)
            response = urllib.request.urlopen(request)
        
        content = response.read()
        if (content):
            try:
                data_json = json.loads(bytes.decode(content))
                #print(data_json)
                #print(data_json['showapi_res_code'])
                if (data_json['showapi_res_code'] != 0):
                    print(str.format('Some Error Happend(JSON RetErr code={},S={},E={})!!\n', code, startdate, enddate))
                    #print(data_json)
                else:
                    self.WriteDataToFile(data_json, code)
                    
            except:
                print(str.format('Some Error Happend(JSON Parse code={},S={},E={},Err={})!!\n', code, startdate, enddate))
                raise
                
        else:
            print('Some Error Happend(No Content)!!\n')

        return

    #------------------------------------------------
    def WriteDataToFile(self, json_data, code):
        path = str.format('D:\\cwf_gupiao\\{}', code)
        
        #make dir
        try:
            os.makedirs(path)
        except:
            #do nothing here
            pass

        #read data got below
        for day_data in json_data['showapi_res_body']['list']:
            v_file_name = str.format('D:\\cwf_gupiao\\{}\\{}.txt', code, day_data['date'])
            
            #if no file exist,new one
            if (os.path.isfile(v_file_name)):
                #we have downloaded it before
                pass
            else:
                v_file = open(v_file_name, 'w')
                #write data to file
                w_data=str.format("code={}\r\nmin_price={}\r\nmax_price={}\r\nopen_price={}\r\nclose_price={}\r\ntrade_money={}\r\ntrade_num={}\r\ndiff_money={}\r\ndiff_rate={}\r\nswing={}\r\ndate={}\r\nturnover={}",
                                 day_data['code'], day_data['min_price'], day_data['max_price'], day_data['open_price'], day_data['close_price'], day_data['trade_money'], day_data['trade_num'],
                                 day_data['diff_money'], day_data['diff_rate'], day_data['swing'], day_data['date'], day_data['turnover'])
                v_file.write(w_data)
                v_file.close()

        return

    #------------------------------------------------
    #get data from startY to today            
    def GetDataByCodeFromStart(self, code, startY=2008, startM=1):
        day=[0,31,28,31,30,31,30,31,31,30,31,30,31]
        int_year = startY
        int_month = startM
        int_day = 1
        sleep_cnt = 0

        today = datetime.date.today()
        print(today)

        while (True):

            startD = str.format("{:0>3}-{:0>2}-{:0>2}", int_year, int_month, int_day)
            endD = str.format("{:0>3}-{:0>2}-{:0>2}", int_year, int_month, day[int_month])

            print(startD)
            print(endD)

            #get data
            self.GetStockDataByCode(code, startD, endD)
            sleep_cnt = sleep_cnt + 1
            
            #it's in future
            if int_year >= today.year and int_month >= today.month and day[int_month] >= today.day:
                print('over')
                break
        
            if int_month >= 12:
                int_month = 1;
                int_year = int_year + 1
            else:
                int_month = int_month + 1;

            if sleep_cnt % 5 == 0:
                print('Sleep 20s\r\n')
                time.sleep(20)
                sleep_cnt = 0

            time.sleep(2)

        return
                      

aa = GetData()
#aa.GetStockDataByCode('600000', '2018-03-01', '2018-03-31')
aa.GetDataByCodeFromStart('600000', 2018, 3)
        
