import pandas as pd

class Covidedit():

    df_kor = pd.read_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/resources/data/kr_daily.csv')
    df_reg = pd.read_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/resources/data/kr_regional_daily.csv')
    del df_kor['released']
    del df_kor['tested']
    del df_kor['negative']
    df_kor.columns =['date','total_cases','total_death']
    df_kor['date']=pd.to_datetime(df_kor['date'].astype(str), format='%Y/%m/%d')
    print(df_kor)

    #print(df_reg)
    df_reg = df_reg[df_reg['region']=='서울']
    #print(df_reg)
    del df_reg['region']
    del df_reg['released']
    df_reg.columns =['date','seoul_cases','seoul_death']
    #print(df_reg)
    df_reg['date']=pd.to_datetime(df_reg['date'].astype(str), format='%Y/%m/%d')
    print(df_reg)
    
    df_all = pd.merge(df_kor,df_reg, on=['date','date'],how='left')
    df_all = df_all.fillna(0)
    df_all['seoul_cases'] = df_all['seoul_cases'].astype(int)
    df_all['seoul_death'] = df_all['seoul_death'].astype(int)
    print(df_all)
    df_all.to_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/resources/data/kor&seoul.csv',encoding='UTF-8')
    # df_all = pd.merge(df_kor, df_reg, on=['date','date'], how='left')

    # df_all['total_cases'] = pd.to_numeric(df_all['total_cases'], errors='coerce').fillna(0).astype(int)
    # df_all['total_deaths'] = pd.to_numeric(df_all['total_deaths'], errors='coerce').fillna(0).astype(int)
    # df_all['ca_cases'] = pd.to_numeric(df_all['ca_cases'], errors='coerce').fillna(0).astype(int)
    # df_all['ca_deaths'] = pd.to_numeric(df_all['ca_deaths'], errors='coerce').fillna(0).astype(int)

    # df_all.to_csv(path+"/covid.csv")
    # print(df_all.head())

