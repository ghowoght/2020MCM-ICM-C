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

# 求声誉-时间曲线
score = []
# 时间窗
days = 1500
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
xobs = np.arange(0, len(s), 1)
xobs = xobs.reshape(-1, 1)
yobs = np.array(s)
yobs = yobs.reshape(-1, 1)
# 拟合
reg.fit(xobs, yobs)
# 预测
means, sigmas = reg.predict(xobs, return_std=True)

# LDA建模
import nltk
from nltk import FreqDist
nltk.download('stopwords')
# 修改时间段、以及对星级进行筛选
df = data.loc[data.review_date>=pd.to_datetime('1/1/2008')]
df = df.loc[data.review_date<=pd.to_datetime('1/7/2010')]
review = df.loc[data.star_rating==1].review_body
# 删除标点和数字
for i in review:
    i = i.replace("[^a-zA-Z#<>]", " ")                
# 删除停止符
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new
review = review.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
review = [remove_stopwords(r.split()) for r in review]
# 进一步去除文本噪音
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output
tokenized_reviews = pd.Series(review).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))
review = reviews_3
# 提取所有单词
list_data = []
for i in review:
    list_data += i.lower().split()                    
# 词频统计
fdist = FreqDist(list_data)
words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
# 词频可视化
d = words_df.nlargest(columns="count", n = 20) 
plt.figure(figsize=(15,5))
ax = sns.barplot(data=d, x= "word", y = "count")
ax.set(ylabel = 'Count')
plt.show()
# LDA主题建模
import pyLDAvis
import pyLDAvis.gensim
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(reviews_2) 
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
LDA = gensim.models.ldamodel.LdaModel
# 开始训练
lda_model = LDA(corpus=doc_term_matrix,
               id2word=dictionary,
               num_topics=5, 
               random_state=100,
               chunksize=1000,
               passes=50)
# 打印结果
for i in lda_model.print_topics():
    print(i)
# 可视化    
#vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
#pyLDAvis.display(vis)
#pyLDAvis.show(vis)
    

## 加权系数
#result_list=[]
#for i in lda_model.print_topics():
#    temp = str(i[1])
#    temp = temp.replace(chr(34), " ")
#    temp = temp.replace("*", " ")
#    temp = temp.replace("+", " ")
#    temp = temp.split()
#    dic = {}
#    j = 0
#    for j in range(int(len(temp) / 2)):
#        dic.update({temp[j * 2 + 1]: float(temp[j * 2])})
#    result_list.append(dic)
##    print(temp)
#result = result.T
#df = pd.DataFrame(result)
#df = pd.concat([df, data['star_rating']],axis=1)
#df.to_excel('result.xlsx', index=False, header=None)
    
# 加权系数
result_list=[]
for i in lda_model.print_topics():
    temp = str(i[1])
    temp = temp.replace(chr(34), " ")
    temp = temp.replace("*", " ")
    temp = temp.replace("+", " ")
    temp = temp.split()
    list0 = []
    j = 0
    for j in range(int(len(temp) / 2)):
        list0+=[temp[j * 2 + 1]]
    result_list.append(list0)

df = pd.DataFrame(result_list)
df.to_excel('result_2c_上升期.xlsx', index=False, header=None)


