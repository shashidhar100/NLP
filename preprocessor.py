import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

class Preprocessor:
    def __init__(self,name="LR_twitter"):
        self.name = name
        self.name_map_dic = {
                              "LR_twitter": LR_Twitter  
                            }
    
    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)


class LR_Twitter:
    def __init__(self,**kwargs):
        self.stemmer = PorterStemmer()
        self.stopwords_english = stopwords.words('english')
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                reduce_len=True)
        self.freqs = self.build_freqs(kwargs["train_x"],kwargs["train_y"])
    
    def process_tweet(self,tweet):
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
            if (word not in self.stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                # tweets_clean.append(word)
                stem_word = self.stemmer.stem(word)  # stemming word
                tweets_clean.append(stem_word)

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
            for word in self.process_tweet(tweet):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1

        return freqs
    
    def extract_features(self,tweet,freqs):
        '''
        Input: 
            tweet: a list of words for one tweet
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        Output: 
            x: a feature vector of dimension (1,3)
        '''
        # process_tweet tokenizes, stems, and removes stopwords
        word_l = self.process_tweet(tweet)
        x = np.zeros((1, 2)) 
        for word in word_l:
            x[0,0] += freqs.get((word, 1.0),0)
            x[0,1] += freqs.get((word, 0.0),0)
            
        return x
    
    def __call__(self,x,y,**kwargs):
        features = self.extract_features(x,self.freqs)
        return features,y
    
    
    def __str__(self):
        return "Logistic Regression Preprocessor for Twitter dataset"

        
    
        
    