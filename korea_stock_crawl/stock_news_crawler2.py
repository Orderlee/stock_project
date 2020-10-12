from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

stock_code = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
                       header=0)[0]
stock_code.종목코드=stock_code.종목코드.map('{:06d}'.format)

stock_code=stock_code[['회사명','종목코드']]

stock_code=stock_code.rename(columns={'회사명':'company','종목코드':'code'})
stock_code.to_csv('/Users/YoungWoo/stock/company.csv', index=False, encoding='UTF-8')
#code_df.head()
stock_code.head()



plusUrl = input('회사:').upper()
plusUrl = stock_code[stock_code.company==plusUrl].code.values[0].strip()

article_result =[]
title_result = []
link_result = []
date_result = []
source_result = []

for i in range(1,3):
    url = 'https://finance.naver.com/item/news_news.nhn?code='+ str(plusUrl)+'&page={}'.format(i)
    source_code = requests.get(url).text
    html = BeautifulSoup(source_code, "lxml")
    rprts = html.findAll("table", {"class":"type5"})

    for items in rprts:
        
        titles = items.select(".title")
        print(titles)
        #title_result=[]
        for title in titles: 
            title = title.get_text() 
            title = re.sub('\n','',title)
            title_result.append(title)

        article = items.select('.title')
        #article_result =[]
        #print(article_result)
        for li in article:
            lis =  'https://finance.naver.com' + li.find('a')['href']
            articles_code = requests.get(lis).text
            htmls = BeautifulSoup(articles_code,"lxml")
            #docs = htmls.find("table",{"class":"view"})
            docs = htmls.find("div",{"class":"scr01"})
            docs = docs.get_text()
            article_result.append(docs)

        links = items.select('.title') 
        #link_result =[]
        print(link_result)
        for link in links: 
            add = 'https://finance.naver.com' + link.find('a')['href']
            print(add)
            link_result.append(add)

        dates = items.select('.date') 
        #date_result = [date.get_text() for date in dates] 
        for date in dates:
            date = date.get_text()
            date_result.append(date)
        #print(date_result)

        sources = items.select('.info')
        #source_result = [source.get_text() for source in sources]
        for source in sources:
            source = source.get_text()
            source_result.append(source)
        #print(source_result)


result= {"날짜" : date_result, "언론사" : source_result, "기사제목" : title_result, "기사내용" : article_result, "링크" : link_result} 
df_result = pd.DataFrame(result)
print(df_result)
            
print("다운 받고 있습니다------")
df_result.to_csv(str(plusUrl)+ '.csv', encoding='utf-8-sig') 
