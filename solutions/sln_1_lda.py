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

#categories = np.unique(data['product_parent'])
#
#colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
#for i, category in enumerate(categories):
#    plt.scatter(data.loc[data.product_parent == category].review_date,
#            data.loc[data.product_parent == category].star_rating,              
#             label=str(category))
#plt.show()
#print(data.star_rating.mean(axis=0))

import nltk
from nltk import FreqDist
nltk.download('stopwords')

#review = data.review_headline
review = data.review_body


# 求评论的有用比例、是否vine、是否验证购买
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


review_length = np.zeros(len(review))
lastest_date = max(data['review_date'])
review_days = np.zeros(len(review))
for i in range(len(data)):
    review_length[i] = len(review[i].lower().split())
    date_diff = lastest_date - data.review_date[i]
    review_days[i] = date_diff.days

# 获取该产品销售数量
categories = np.unique(data['product_parent'])
total_sell_dict = {}
total_sell = np.zeros(len(data))
for i, category in enumerate(categories):
    t = {category:len(data.loc[data.product_parent == category])}
    total_sell_dict = {**total_sell_dict, **t} 
for i in range(len(review)):
    total_sell[i] = total_sell_dict[data.product_parent[i]]
    


df = pd.concat([pd.DataFrame(star_rating, columns=['star_rating']),
                pd.DataFrame(vine, columns=['vine']),
                pd.DataFrame(verified_purchase, columns=['verified_purchase']),
                pd.DataFrame(review_length, columns=['review_length']),
                pd.DataFrame(review_days, columns=['review_days']),
                pd.DataFrame(total_sell, columns=['total_sell']),
                pd.DataFrame(helpful_votes, columns=['helpful_votes']),
                pd.DataFrame(helpfulness_rating, columns=['helpfulness_rating'])
                ],axis=1)
df.to_excel('data_ques_1.xlsx', index=False)

df = df.loc[df.helpful_votes!=0]
    
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

corr = df.corr()


#sns.pairplot(df, x_vars=['star_rating','vine','verified_purchase','review_length','review_days','total_sell'], 
#             y_vars='helpful_votes', size=7, aspect=0.8,kind = 'reg')
#plt.savefig('各变量回归关系.jpg')

helpful_review_index=pd.read_csv("helpful_review_index.csv")

review=np.array(review)
helpful_review_index=np.array(helpful_review_index)

df=pd.concat([pd.DataFrame(review,columns=['review']),
              pd.DataFrame(helpful_review_index,columns=['helpful_review_index'])
              ],axis=1)
df=df.loc[df.helpful_review_index==1]


# LDA建模
review=df.review
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

# LDA主题建模
import pyLDAvis
import pyLDAvis.gensim
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(reviews_2) 
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
LDA = gensim.models.ldamodel.LdaModel
# 确定主题数量

# 开始训练
lda_model = LDA(corpus=doc_term_matrix,
               id2word=dictionary,
               num_topics=10, 
               random_state=100,
               chunksize=1000,
               passes=50)

for i in lda_model.print_topics():
    print(i)
    
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.display(vis)
pyLDAvis.show(vis)

