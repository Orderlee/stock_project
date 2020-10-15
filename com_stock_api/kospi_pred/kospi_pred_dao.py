import mysql.connector
from com_stock_api.ext.db import config

class KospiDao:

    def __init__(self):
        self.connector = mysql.connector.cursor(dictionary=True)

    def select_kospi(self):
        cur = self.cursor
        con = self.connector
        rows = []

        try:
            cur.execute('select * from kospi_pred',)
            rows = cur.fetchall()
            for row in rows:
                print(f'price is : {str(row["price"])}')

            cur.close()

        except:
            print('Exception ...')

        finally:
            if con is not None:
                con.close()

        return rows

print('----2-----')
dao = KospiDao()
dao.select_kospi()