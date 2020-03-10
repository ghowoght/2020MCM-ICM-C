# 统计产品数量和平均星级

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import optimize

#data = pd.read_csv("macrowave.csv")
#data = pd.read_csv("pacifier.csv")
data = pd.read_csv("hair_dryer.csv")
data = data.loc[data.review_date != '']
data.review_date=pd.to_datetime(data.review_date)

data = data.loc[~pd.isna(data.review_date)]

categories = np.unique(data['product_parent'])
#
#colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
#for i, category in enumerate(categories):
#    plt.scatter(data.loc[data.product_parent == category].review_date,
#            data.loc[data.product_parent == category].star_rating,              
#             label=str(category))
#plt.show()
#print(data.star_rating.mean(axis=0))

list=[]
for i, category in enumerate(categories):
    list.append([category,
                 data.loc[data.product_parent == category].star_rating.mean(axis=0),
                 len(data.loc[data.product_parent == category])])


df = pd.DataFrame(list, columns=['product_parent', 'star_rating', 'num'])
df=df.sort_values(by='num',axis=0,ascending=0)
