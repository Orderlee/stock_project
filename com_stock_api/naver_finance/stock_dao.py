import mysql.connector
from com_stock_api.ext.db import config

class StockDao:

    def __init__(self):
        self.connector = mysql.connector.connect(**config)
        self.cursor = self.connector.cursor(dictionary=True)

    def select_stocks(self):
        cur = self.cursor
        con = self.connector
        rows = []

        try:
            cur.execute('select * from naver_finance',)
            rows = cur.fetchall()
            for row in rows:
                print(f'price is : {str(row["close"])}')
            cur.close()
        except:
            print('Exception ...')

        finally:
            if con is not None:
                con.close()
        return rows

print('----2-----')
dao = StockDao()
dao.select_stocks()