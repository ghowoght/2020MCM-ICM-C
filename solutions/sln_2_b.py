import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import optimize
import spacy

#data = pd.read_csv("macrowave.csv")
#data = pd.read_csv("pacifier.csv")
data = pd.read_csv("hair_dryer.csv")
data = data.loc[data.review_date != '']
data.review_date=pd.to_datetime(data.review_date)

data = data.loc[~pd.isna(data.review_date)]

review_date = np.unique(data['review_date'])
date_range = pd.date_range(start=review_date[0], end=review_date[len(review_date)-1])

# 求时间趋势
score = []
# 时间窗
days = 1000
for i in range(days,len(date_range),1):
    date1 = date_range[i]
    date0 = date_range[i-days]
    df = data.loc[data.review_date>=date0]
    if(len(df.loc[data.review_date<=date1])>0):
        score.append((df.loc[data.review_date<=date1].star_rating.mean()))
#                    *len(df.loc[data.review_date<=date1])/len(data))
    else:
        score.append(0)
plt.plot(date_range[days:],score)
plt.show()
#score = pd.DataFrame(score,columns=['score'])


# 使用GRP进行曲线拟合
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C# REF就是高斯核函数

#核函数的取值
kernel = C(1.0, (1e-3, 1e3))*RBF(0.5, (1e-4,10))
#创建高斯过程回归,并训练
reg = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1)

# 重采样
t = pd.Series(score,index=date_range[days:])
s=t.resample('M').mean()

#xobs = np.array(date_range[days:])
xobs = np.arange(0, len(s), 1)
xobs = xobs.reshape(-1, 1)

yobs = np.array(s)
yobs = yobs.reshape(-1, 1)

reg.fit(xobs, yobs)

means, sigmas = reg.predict(xobs, return_std=True)

plt.figure(figsize=(8, 5))
plt.errorbar(s.index, means, yerr=sigmas, alpha=0.5)
plt.plot(s.index, means, 'g', linewidth=4)

plt.show()


# 平稳性检测
df = pd.concat([pd.DataFrame(date_range[days:],columns=['date']),
                pd.DataFrame(score,columns=['score'])],
                axis=1)

from statsmodels.tsa import stattools
from statsmodels.stats.diagnostic import unitroot_adf

print(unitroot_adf(df.score))

#plt.stem(stattools.acf(df.score));
k = stattools.pacf(df.score)
k[1] = 0.003
plt.stem(k);

# 画出波峰波谷
data = means.reshape((len(means),))
doublediff = np.diff(np.sign(np.diff(data)))
peak_locations = np.where(doublediff == -2)[0] + 1
peak_locations = peak_locations[:(len(peak_locations)-1)]

doublediff2 = np.diff(np.sign(np.diff(-1*data)))
trough_locations = np.where(doublediff2 == -2)[0] + 1

#product_name = 'Macrowave\'s'
#product_name = 'Pacifier\'s'
product_name = 'Hair-Dryer\'s'
# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
#plt.plot('date', 'traffic', data=s, color='tab:blue', label='Air Traffic')
plt.errorbar(s.index, data, yerr=sigmas, alpha=0.5, label='90% confidence intervals')
plt.plot(s.index, data, 'b', linewidth=1, label=product_name+' Reputation')
plt.scatter(s.index[peak_locations], data[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
plt.scatter(s.index[trough_locations], data[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')

# Annotate
date =  s.index.map(lambda x: str(x.year) + '-' +  str(x.month))
for t, p in zip(trough_locations, peak_locations):
    plt.text(s.index[p], data[p] - 0.05, date[p], horizontalalignment='center', color='darkgreen', fontsize=16)
    plt.text(s.index[t], data[t] + 0.05, date[t], horizontalalignment='center', color='darkred', fontsize=16)

# Decoration
#plt.ylim(2.5,4)
#xtick_location = s.index.tolist()[::6]
#xtick_labels = s.index.tolist()[::6]
#plt.xticks(labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
plt.title('Peak and Troughs of '+product_name+' Reputation(2008-2015)', fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
plt.xticks(fontsize=18, alpha=.7)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)

plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
plt.show()

