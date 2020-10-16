import mysql.connector
from com_stock_api.ext.db import config
from com_stock_api.korea_covid.korea_covid_dao import KoreaDao
from com_stock_api.kospi_pred.kospi_pred_dao import KospiDao
from com_stock_api.naver_finance.stock_dao import StockDao
from com_stock_api.naver_news.news_dao import NewsDao

class HomeDao():

    def __init__(self):
        self.korea_covid_dao = KoreaDao()
        self.kospi_pred_dao = KospiDao()
        self.stock_dao = StockDao()
        self.news_dao = NewsDao()

    def create_tables(self):
        ...
        
