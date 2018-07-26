import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report




plt.style.use('fivethirtyeight')


df = pd.read_csv('clean.csv')
df.columns = ["id","text","sentiment"]
df.drop(['id'], axis=1, inplace=True)
print(df.head())

vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = True,
    stop_words = 'english',
)



corpus_data_features = vectorizer.fit_transform(df['text'].values.astype('U'))

corpus_data_features_nd = corpus_data_features.toarray()
corpus_data_features_nd.shape

vocab = vectorizer.get_feature_names()
print vocab

X_train, X_test, y_train, y_test  = train_test_split(
    corpus_data_features_nd[0:len(df)], 
    df.sentiment,
    train_size=0.80, 
    random_state=1234)


log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)


y_pred = log_model.predict(X_test)

print(classification_report(y_test, y_pred))



""" tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}

neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b') """





def tweet_cleaner_updated(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed == souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()



""" df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None,
                 usecols=[0,5],names=['sentiment','text'])
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
df.head()

print "Cleaning the tweets...\n"
clean_tweet_texts = []
for i in xrange(0,len(df)):
    if( (i+1)%1000 == 0 ):
        print "Tweets %d of %d has been processed" % ( i+1, len(df) )                                                                    
    clean_tweet_texts.append(tweet_cleaner_updated(df['text'][i]))
    clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])


clean_df['target'] = df.sentiment
clean_df.to_csv('clean_tweet.csv',encoding='utf-8')

csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()


my_df.info() """