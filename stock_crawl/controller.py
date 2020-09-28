import sqlite3
from bs4 import BeautifulSoup
from stock_crawl.entity import Entity
from stock_crawl.service import Service

class Controller:
    def __init__(self):
        self.entity = Entity()
        self.service = Service()

    def StockState():
        