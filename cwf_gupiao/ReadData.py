import sys
import os
import datetime
import time


class ReadData:
    m_list_stock_data=[]
    #--------------------------------------------
    def ReadStockData(self, path):
        flist = os.listdir(path)

        for i in range(0, len(flist)):
            m_stockdata={}
            fpath = os.path.join(path, flist[i])
            if os.path.isfile(fpath):
                #print(fpath)
                v_file = open(fpath, 'r')
                v_lines = v_file.readlines()

                for v_line in v_lines:
                    v_line = v_line.splitlines()
                    list_data = v_line[0].split('=', 2)
                    if len(list_data) >= 2 and list_data[0] != '' and list_data[1] != '':
                        m_stockdata[list_data[0]] = list_data[1]
                    else:
                        #print(str.format('ReadStockData Err path={}',fpath))
                        pass
                
            self.m_list_stock_data.append(m_stockdata)

        #sort by date
        self.m_list_stock_data.sort(key=lambda data_e: data_e['date'], reverse=False)
        return
        


#bb=ReadData()
#bb.ReadStockData('D:\\cwf_gupiao\\600000\\')
#print(bb.m_list_stock_data[0]['date'])
#print(bb.m_list_stock_data[1]['date'])
#print(bb.m_list_stock_data[2]['date'])
#print(bb.m_list_stock_data[3]['date'])
