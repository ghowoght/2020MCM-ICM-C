import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import optimize
import spacy

data = pd.read_csv("macrowave.csv")
#data = pd.read_csv("pacifier.csv")
#data = pd.read_csv("hair_dryer.csv")
data = data.loc[data.review_date != '']
data.review_date=pd.to_datetime(data.review_date)

data = data.loc[~pd.isna(data.review_date)]

review_date = np.unique(data['review_date'])
date_range = pd.date_range(start=review_date[0], end=review_date[len(review_date)-1])

# 求时间趋势
score = []
# 时间窗
days = 365
for i in range(days,len(date_range),1):
    date1 = date_range[i]
    date0 = date_range[i-days]
    df = data.loc[data.review_date>=date0]
    score.append(df.loc[data.review_date<=date1].star_rating.mean())
#plt.plot(date_range[days:],score)
#plt.show()

# 求评论的有用比例
helpfulness_rating = np.zeros(len(data))
vine = np.zeros(len(data))
verified_purchase = np.zeros(len(data))
for i in range(len(data)):
    if(data.total_votes[i]!=0):
        helpfulness_rating[i] = data.helpful_votes[i]/data.total_votes[i]
    if(data.vine[i]=='Y'):
        vine[i]=1
    if(data.verified_purchase[i]=='Y'):
        verified_purchase[i]=1

helpful_votes = np.array(data.helpful_votes)
star_rating = np.array(data.star_rating)

df1 = pd.DataFrame(star_rating[days:])
df2 = pd.DataFrame(helpful_votes[days:])
df3 = pd.DataFrame(helpfulness_rating[days:])
df4 = pd.DataFrame(vine[days:])
df5 = pd.DataFrame(verified_purchase[days:]) 
#df6 = pd.DataFrame(score)
df = pd.concat([df1, df2, df3, df4, df5],axis=1)
df.to_excel('result_2_c.xlsx', index=False, header=None)

