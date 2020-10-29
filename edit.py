
# import matplotlib.pyplot as plt
# %matplotlib inline
# df_stock['close'].plot(figsize=(12,6), grid=True)

# # 반응형 차트 그리는 코드
# fig = px.line(df, x='date', y='close', title='{}의 종가(close) Time Series'.format(company))

# fig.update_xaxes(
#     rangeslider_visible=True,
#     rangeselector=dict(
#         buttons=list([
#             dict(count=1, label="1m", step="month", stepmode="backward"),
#             dict(count=3, label="3m", step="month", stepmode="backward"),
#             dict(count=6, label="6m", step="month", stepmode="backward"),
#             dict(step="all")
#         ])
#     )
# )
# fig.show()


# #단순형 차트
# import matplotlib.pyplot as plt
# # 필요한 모듈 import 하기 
# import plotly
# import plotly.graph_objects as go
# import plotly.express as px

# # %matplotlib inline 은 jupyter notebook 사용자용 - jupyter notebook 내에 그래프가 그려지게 한다.
# %matplotlib inline 

# plt.figure(figsize=(10,4))
# plt.plot(df['date'], df['close'])
# plt.xlabel('')
# plt.ylabel('close')
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# plt.savefig(company + ".png")
# plt.show()






# # -*- coding: utf-8 -*- 
# import os
# basedir = os.path.abspath(os.path.dirname(__file__))
# import numpy as np
# import pandas as pd
# from pandas import read_table
# from com_stock_api.utils.file_helper import FileReader
# from com_stock_api.utils.checker import is_number
# from collections import defaultdict
# import math
# from com_stock_api.naver_news.service_backup import NewsService

# class NewsAnalysis:
#     def __init__(self,k = 0.5):
#         self.k = k
#         self.data = os.path.abspath(__file__+"/.."+"/data/")
#         #print(self.data) -> /Users/YoungWoo/stock_psychic_api/com_stock_api/naver_news/data
#         self.reader =FileReader()


#     def train(self):
#         training_set = self.load_corpus()
#         # 범주 0 (긍정), 범주 1 (부정) 문서의 수를 세어줌
#         num_class0 = len([1 for _, point in training_set if point > 3.5])
#         num_class1 = len(training_set) - num_class0
#         word_counts = self.count_words(training_set)
#         self.word_probs = self.word_probabilities(word_counts, num_class0, num_class1, self.k)

    

#     def load_corpus(self):
#         reader =self.reader
#         path = self.data
#         corpus = read_table(path + '/movie_review.csv', sep=',',encoding='UTF-8')
#         #print(f'Corpus Spec : {corpus}')
#         return np.array(corpus)
    
#     def count_words(self, traing_set):
#         counts = defaultdict(lambda: [0,0])
#         for doc, point in traing_set:
#             # 영화리뷰가 test일때만 카운팅
#             if is_number(doc) is False:
#                 words = doc.split()
#                 for word in words:
#                     counts[word][0 if point > 3.5 else 1] += 1
#         return counts
    
#     def word_probabilities(self, counts, total_class0, total_class1, k):
#         # 단어의 빈도수를 [단어,p(w|긍정), p(w|부정)] 형태로 전환
#         return [(W,
#         (class0 + k) / (total_class0 + 2 * k), 
#         (class1 + k) / (total_class1 + 2 * k))
#         for W, (class0, class1) in counts.items()]

#     def class0_probability(self,word_probs,doc):
#         # 별도의 토크나이즈 하지 않고 띄어쓰기 
#         docwords = doc.split()
#         log_prob_if_class0 = log_prob_if_class1 = 0.0
#         #모든단어에 반복
#         for word, prob_if_class0, prob_if_class1 in word_probs:
#             # 만약 리뷰에 word가 나타나면 해당 단어가 나올 log에 확률을 더해줌
#             if word in docwords:
#                 log_prob_if_class0 += math.log(prob_if_class0)
#                 log_prob_if_class1 += math.log(prob_if_class1)
#                 # 만약 리뷰에 word 가 없으면 해당 단어가 안나올 log에 확률을 더해줌
#                 # 나오지 않을 확률은 log ( 1 - 나올 확률) 로 계산
#             else:
#                 log_prob_if_class0 += math.log(1.0 - prob_if_class0)
#                 log_prob_if_class1 += math.log(1.0 - prob_if_class1)

#             prob_if_class0 = math.exp(log_prob_if_class0)
#             prob_if_class1 = math.exp(log_prob_if_class1)

#             return prob_if_class0 / (prob_if_class0 + prob_if_class1)

  

#     def classify(self,doc):
#         return self.class0_probability(self.word_probs, doc)

        

#     def hook(self,txt):
#         print('====hook====')
#         self.train()
#         return self.classify(txt)
        

#     def makelabel(self):
#         path = self.data
#         # 1. 수집 
#         # 2. 모델
#         #service = NewsService()
#         #service.new_model()
#         # 3. CRUD
#         df_result = pd.read_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/resources/data/011070.csv',encoding='utf-8', dtype=str)
#         # 4. Eval
        

#         df_result['label']= 0.0
#         for i in range(0,2500):
#             try:
#                 df_result['label'][i] = '%.2f' % self.hook(df_result['content'][i])
#             except KeyError : 
#                 pass
        
#         df_result.to_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/naver_news/data/lginnotek1.csv',encoding='UTF-8')

#         return df_result

# if __name__=='__main__':
#     service = NewsAnalysis()
#     service.makelabel()