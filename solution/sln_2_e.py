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

import nltk
from nltk import FreqDist
nltk.download('stopwords')

#review = data.review_headline
review = data.review_body

# 删除标点和数字
for i in range(len(review)):
    review[i] = review[i].replace("[^a-zA-Z]", " ")
                
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

list_data = []
for i in review:
    list_data += i.lower().split()
                          
# 词频统计
fdist = FreqDist(list_data)
words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

d = words_df.nlargest(columns="count", n = 30) 
plt.figure(figsize=(20,5))
ax = sns.barplot(data=d, x= "word", y = "count")
ax.set(ylabel = 'Count')
plt.show()


adj_list=[]
nlp = spacy.load('en_core_web_sm')
for i in range(len(words_df)):
    doc = nlp(words_df.word[i])
    pos = [token.pos_ for token in doc]
    if(pos==['ADJ']):
        adj_list.append(words_df.word[i])

# 形容词 词频统计
adj_num = []
for i in range(len(adj_list)):
    adj_num.append(fdist[adj_list[i]])
adj_words_df = pd.DataFrame({'word':adj_list, 'count':adj_num})

#adj_words_df.to_excel('adj_words.xlsx')

largest_word = adj_words_df.nlargest(columns="count", n = 30) 
plt.figure(figsize=(20,5))
ax = sns.barplot(data=largest_word, x= "word", y = "count")
ax.set(ylabel = 'Count')
plt.show()

largest_word = np.array(largest_word.word,dtype=str)
result = np.zeros((len(largest_word),len(review)))
for i in range(len(largest_word)):
    term = largest_word[i]
    j = 0
    for rev in review:
        if(rev.find(term) > 0):
            result[i][j] = 1
        j += 1

result = result.T
largest_word_list = []
for i in largest_word:
    largest_word_list.append(str(i))
df = pd.DataFrame(result, columns=largest_word_list)
df = pd.concat([df, data['star_rating']],axis=1)
df.to_excel('data_2_e.xlsx', index=False)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

corr = df.corr()

sns.pairplot(df, x_vars='great',#largest_word_list[:10],
             y_vars='star_rating', size=7, aspect=0.8,kind = 'reg')
#sns.pairplot(df, x_vars=largest_word_list[10:20],#df.iloc[0], 
#             y_vars='star_rating', size=7, aspect=0.8,kind = 'reg')
#sns.pairplot(df, x_vars=largest_word_list[20:30],#df.iloc[0], 
#             y_vars='star_rating', size=7, aspect=0.8,kind = 'reg')
plt.savefig('各变量回归关系.jpg')
plt.show()

special_word_freq = np.zeros(len(data))
i = 0
special_word = 'enthusiastic'
#special_word = 'disappointed'
for rev in review:
    if(rev.find(special_word) > 0):
        special_word_freq[i] = 1
    i += 1
special_word_freq = pd.concat([pd.DataFrame(special_word_freq,columns=[special_word]),
                               data['star_rating']],axis=1)
sns.pairplot(special_word_freq, x_vars=special_word,
             y_vars='star_rating', size=7, aspect=0.8,kind = 'reg')

plt.show()
