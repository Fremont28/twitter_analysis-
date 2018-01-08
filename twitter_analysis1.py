#import libraries 
from TwitterSearch import *
#1. search through the tweets
tso = TwitterSearchOrder()
tso.set_keywords(['#Milan'])

ts = TwitterSearch(
            consumer_key = 'aSIenX8nyf75Lx3no4j5xuw3a',
            consumer_secret = 'dzQPbwEGcGfTB0uSRuRWxaHxbRpGr68jpS8SNG2lk2FFUtRc3o',
            access_token = '42704027-Aj1hOUheQ0awUF1Bepayab2G8q57mOvyPIuWj4YSl',
            access_token_secret = 'Ab2MS9fSqybmS6t6BM6fQojEKjU72PPjkwz81iDbZpMs9'
        )
tweets_hashtag=[]
for tweet in ts.search_tweets_iterable(tso):
    tweets_hashtag.append(tweet)
    print(tweet['user']['screen_name'],tweet['text'])

type(tweets_hashtag) #type list

#a. advanced filtering
# from TwitterSearch import *
from TwitterSearch import *
tso=TwitterSearchOrder()
tso.set_keywords(['Fiat','Milan'])

ts=TwitterSearch(
        consumer_key='aSIenX8nyf75Lx3no4j5xuw3a',
        consumer_secret='dzQPbwEGcGfTB0uSRuRWxaHxbRpGr68jpS8SNG2lk2FFUtRc3o',
        access_token='42704027-Aj1hOUheQ0awUF1Bepayab2G8q57mOvyPIuWj4YSl',
        access_token_secret='Ab2MS9fSqybmS6t6BM6fQojEKjU72PPjkwz81iDbZpMs9'
    )
tweets_type=[]
for tweet in ts.search_tweets_iterable(tso):
    print(tweet['user']['screen_name'],tweet['text'])
    tweets_type.append(tweet) #this works 

#2. search specific user 
tuo=TwitterUserOrder('nytimes')

#create tso
ts = TwitterSearch(
            consumer_key = 'aSIenX8nyf75Lx3no4j5xuw3a',
            consumer_secret = 'dzQPbwEGcGfTB0uSRuRWxaHxbRpGr68jpS8SNG2lk2FFUtRc3o',
            access_token = '42704027-Aj1hOUheQ0awUF1Bepayab2G8q57mOvyPIuWj4YSl',
            access_token_secret = 'Ab2MS9fSqybmS6t6BM6fQojEKjU72PPjkwz81iDbZpMs9'
        )

#user timeline
tweets_peach=[]
for tweet in ts.search_tweets_iterable(tuo):
    tweets_peach.append(tweet)
type(tweets_peach) #list 

#extracting a list
parse1=[tweets_peach[i] for i in (0,1)]
type(parse1)

#convert list to string
import json 
json_string=json.dumps(tweets_peach)
type(json_string) #string
json_string1=json.loads(json_string)
type(json_string1) #list 
str1=" ".join(str(x) for x in tweets_peach) #create a string source: https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string
type(str1)

#find most common words
words=re.findall(r'\w+',str1.lower())
Counter(words).most_common(50)

#another most common word method 
import re
import string 
frequency={}
match_pattern = re.findall(r'\b[a-z]{3,15}\b',json_string)

for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1


frequency_list=frequency.keys()
frequency_list 
type(frequency_list)

# (LDA)
#clean and preprocess
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop=set(stopwords.words('english'))
exclude=set(string.punctuation)
lemma=WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

str2=[clean(doc).split() for doc in frequency_list]

#DTM 
import gensim 
from gensim import corpora 

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
dictionary = corpora.Dictionary(str2)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in str2]

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in str2]

#lda model
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=10, num_words=4)) #works 


#(LDA and NMF)
documents=frequency_list 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#NMF is able to use tfidf
no_features=500
tfidf_vectorizer=TfidfVectorizer(max_df=0.95,min_df=1,max_features=no_features,stop_words='english')
tfidf=tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names=tfidf_vectorizer.get_feature_names()

# running NMF
from sklearn.decomposition import NMF, LatentDirichletAllocation
no_topics=20
nmf=NMF(n_components=no_topics,random_state=1,alpha=0.1,l1_ratio=0.5,init='nndsvd').fit(tfidf)
nmf 

#display results
def display_topics(model,feature_names,no_top_words):
    for topic_idx,topic in enumerate(model.components_):
        print(topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words=10
display_topics(nmf,tfidf_feature_names,no_top_words) 


#print out common words
tweet_freq=[]
for x in frequency_list:
    print(x,frequency[x])
    tweet_freq.append((x,frequency[x]))

















































#list to str source: https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string
str_text=''.join(str(e) for e in tweets_peach)
type(str_text)

###tokenize
str_text1=str_text.lower()
len(str_text1)








