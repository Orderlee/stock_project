import mysql.connector
from com_stock_api.ext.db import config

class KoreaDao:

    def __init__(self):
        self.connector = mysql.connector.connect(**config)
        self.cursor = self.connector.cursor(dictionary=True)

    def select_covid(self):
        cur = self.cursor
        con = self.connector
        rows = []

        try:
            cur.execute('select * from korea_covid',)
            rows = cur.fetchall()
            for row in rows:
                print(f'total death is : {str(row["total death"])}')
            cur.close()

        except:
            print('Exception ...')

        finally:
            if con is not None:
                con.close()

        return rows


print('---2---')
dao = KoreaDao
dao.select_covid()