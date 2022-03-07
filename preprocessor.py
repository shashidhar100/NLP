import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from torch import le
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class Preprocessor:
    def __init__(self,name="LR_twitter"):
        self.name_map_dic = {
                              "LR_twitter": LR_Twitter  
                            }
        name_list = list(self.name_map_dic.keys())
        if name not in name_list:
            raise ValueError("Preprocessor name should be one of these {}".format(name_list))
        self.name = name

    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)


class LR_Twitter:
    def __init__(self,
                train_x=None,
                train_y=None,
                rm_stop = False,
                rm_punc = False,
                lema = True,
                stem = False,
                n_gram_feat = False,
                n_gram_method = None,
                **kwargs
                ):
        self.rm_stop = rm_stop
        self.rm_punc = rm_punc
        self.lema = lema
        self.stem = stem
        self.n_gram_feat = n_gram_feat
        self.n_gram_method = n_gram_method
        if n_gram_feat:
            self.n_gram_methods_map = {"bog":CountVectorizer,"tfidf":TfidfVectorizer}
            if self.n_gram_method == None:
                self.n_gram_method = "bog"
            self.build_word_matrix(train_x,**kwargs)
        else:
            self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                reduce_len=True)
            if self.lema:
                self.lemmatizer = WordNetLemmatizer()
            if self.stem:
                self.stemmer = PorterStemmer()
            self.freqs = self.build_freqs(train_x,train_y)
            self.stopwords_english = stopwords.words('english')

        
    
    def process_tweet(self,tweet,rm_stop=False,rm_punc=False,stem=False,lema=True):
        """Process tweet function.
        Input:
            tweet: a string containing a tweet
        Output:
            tweets_clean: a list of words containing the processed tweet

        """
        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)
        # tokenize tweets

        tweet_tokens = self.tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if rm_stop:
                if word in self.stopwords_english:
                    continue
            if rm_punc:
                if word in string.punctuation:
                    continue
            if stem:
                word = self.stemmer.stem(word)
            
            if lema and stem==False:
                word = self.lemmatizer.lemmatize(word)
                
            tweets_clean.append(word)

        return tweets_clean


    def build_freqs(self,tweets, ys):
        """Build frequencies.
        Input:
            tweets: a list of tweets
            ys: an m x 1 array with the sentiment label of each tweet
                (either 0 or 1)
        Output:
            freqs: a dictionary mapping each (word, sentiment) pair to its
            frequency
        """
        # Convert np array to list since zip needs an iterable.
        # The squeeze is necessary or the list ends up with one element.
        # Also note that this is just a NOP if ys is already a list.
        yslist = np.squeeze(ys).tolist()

        # Start with an empty dictionary and populate it by looping over all tweets
        # and over all processed words in each tweet.
        freqs = {}
        for y, tweet in zip(yslist, tweets):
            for word in self.process_tweet(tweet,rm_punc=self.rm_punc,rm_stop=self.rm_stop,lema=self.lema,stem=self.stem):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1

        return freqs
    
    def extract_features(self,tweet,freqs,if_bias=False):
        '''
        Input: 
            tweet: a list of words for one tweet
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        Output: 
            x: a feature vector of dimension (1,3) or (1,2)
        '''
        # process_tweet tokenizes, stems, and removes stopwords
        word_l = self.process_tweet(tweet,rm_punc=self.rm_punc,rm_stop=self.rm_stop,lema=self.lema,stem=self.stem)
        if if_bias:
            x = np.zeros((1, 3))
            x[0,0] = 1.0
            for word in word_l:
                x[0,1] += freqs.get((word, 1.0),0)
                x[0,2] += freqs.get((word, 0.0),0)
        else:
            x = np.zeros((1,2))
            for word in word_l:
                x[0,0] += freqs.get((word, 1.0),0)
                x[0,1] += freqs.get((word, 0.0),0)   
        return x

    def build_word_matrix(self,X,**kwargs):
        self.vectorizer = self.n_gram_methods_map[self.n_gram_method](**kwargs).fit(X)
    
    def extract_n_gram_feats(self,x):
        return self.vectorizer.transform(x).toarray()



        

    
    def __call__(self,x,y):
        if self.n_gram_feat:
            return self.extract_n_gram_feats([x]),y
        else:
            return self.extract_features(x,self.freqs),y
    
    
    def __str__(self):
        return "Preprocessor for Twitter dataset"

        
    
        
    