import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.utils.file_helper import FileReader



class NewsService():
    
    def __init__(self):
        self.filereader = FileReader()
        self.datapath = os.path.abspath("com_stock_api/naver_news/")

    def hook(self):
        this = self.filereader
        this.context = os.path.join(basedir, 'data/')
        this.fname = '011070.csv'
        ksco = this.csv_to_dframe()
        print(ksco.head())

if __name__ == '__main__':
    ns = NewsService()
    ns.hook()