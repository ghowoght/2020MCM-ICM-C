# 调用情感分析库进行情感分析，效果不好，正确率在70%以下

#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
#from scipy import optimize
#import spacy
#
#data = pd.read_csv("macrowave.csv")
##data = pd.read_csv("pacifier.csv")
##data = pd.read_csv("hair_dryer.csv")
#data = data.loc[data.review_date != '']
#data.review_date=pd.to_datetime(data.review_date)
#
#data = data.loc[~pd.isna(data.review_date)]
#
#import nltk
#from nltk import FreqDist
#nltk.download('stopwords')
#
##review = data.review_headline
#review = data.review_body
## 删除标点和数字
#for i in review:
#    i = i.replace("[^a-zA-Z#<>]", " ")
#                    
## 删除停止符
#from nltk.corpus import stopwords
#stop_words = stopwords.words('english')
#def remove_stopwords(rev):
#    rev_new = " ".join([i for i in rev if i not in stop_words])
#    return rev_new
#review = review.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
#review = [remove_stopwords(r.split()) for r in review]
#
## 进一步去除文本噪音
#nlp = spacy.load('en', disable=['parser', 'ner'])
#def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
#       output = []
#       for sent in texts:
#             doc = nlp(" ".join(sent)) 
#             output.append([token.lemma_ for token in doc if token.pos_ in tags])
#       return output
#tokenized_reviews = pd.Series(review).apply(lambda x: x.split())
#reviews_2 = lemmatization(tokenized_reviews)
#reviews_3 = []
#for i in range(len(reviews_2)):
#    reviews_3.append(' '.join(reviews_2[i]))
#review = reviews_3
#
#
#nltk.download('vader_lexicon')
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
result=[]
result0=np.zeros(len(review))
i=0
for sentence in review:
#    test(sentence)
    sia = SentimentIntensityAnalyzer()
    ps=sia.polarity_scores(sentence)
    result.append(ps)
    if(ps['compound'] < 0.3):
        result0[i]=-1
    elif(ps['compound'] > 0.0):
        result0[i]=1
    else:
        result0[i]=0
    i+=1
    
result1=np.array(data['star_rating'])
for i in range(len(result1)):
    if(result1[i]>=4):
        result1[i]=1
    elif(result1[i]<=2):
        result1[i]=-1
    else:
        result1[i]=0

df = pd.DataFrame(result0)
df1 = pd.DataFrame(result1)
df = pd.concat([df, df1],axis=1)
cnt=0
for i in range(len(df1)):
    if(abs(result0[i]-result1[i])>=1):
        cnt+=1
print(cnt/len(result0))

df = pd.DataFrame(result)
df = pd.concat([df, df1],axis=1)
df.to_excel('result0.xlsx', index=False, header=None)

        
        
        
        
        