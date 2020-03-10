# 主题建模后将评论映射到主题空间
# 无意义

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

#d = words_df.nlargest(columns="count", n = 30) 
#plt.figure(figsize=(20,5))
#ax = sns.barplot(data=d, x= "word", y = "count")
#ax.set(ylabel = 'Count')
#plt.show()

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
               num_topics=50, 
               random_state=100,
               chunksize=1000,
               passes=50)

for i in lda_model.print_topics():
    print(i)
    

vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
#pyLDAvis.display(vis)
#pyLDAvis.show(vis)



df=pd.DataFrame(vis[1])
categories = np.unique(df['Category'])
result = np.zeros((len(categories),len(review)))
for i, category in enumerate(categories):
    terms = df.loc[df.Category == category].Term
    j = 0
    for rev in review:
        for term in terms:
            if(rev.find(term) > 0):
                result[i][j] += 1
        j += 1

result = result.T
df = pd.DataFrame(result)
df = pd.concat([df, data['star_rating']],axis=1)
df.to_excel('result.xlsx', index=False, header=None)

# 加权系数
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
#        dic.update({temp[j * 2 + 1]: temp[j * 2]})
#    result_list.append(dic)
##    print(temp)
#    
#result = np.zeros((len(result_list),len(review)),dtype=np.float64)
#for i in range(len(result_list)):
#    terms = result_list[i]
#    j = 0
#    for rev in review:
#        for term in terms:
#            if(rev.find(term) > 0):
#                result[i][j] += np.float64(result_list[i][term])
#        j += 1
#    
#result = result.T
#df = pd.DataFrame(result)
#df = pd.concat([df, data['star_rating']],axis=1)
#df.to_excel('result.xlsx', index=False, header=None)
    


