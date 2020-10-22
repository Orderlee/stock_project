import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.util.file_reader import FileReader
import pandas as pd


class CovidService():
    
    def __init__(self):
        self.filereader = FileReader()
        self.datapath = os.path.abspath("com_stock_api/korea_covid/")

    def hook(self):
        this = self.filereader
        this.context = os.path.join(basedir, 'data/')
        this.fname = 'kor&seoul.csv'
        ksco = this.csv_to_dframe()
        print(ksco.head())

if __name__ == '__main__':
    cs = CovidService()
    cs.hook()

    