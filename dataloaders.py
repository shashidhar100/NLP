import nltk
nltk.download("twitter_samples")
from nltk.corpus import twitter_samples
import random
import numpy as np

from torch.utils.data import Dataset

from preprocessor import Preprocessor


class Dataset:
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
        self.dataset_map_dic = {
            "twitter": Twitter
        }
        name_list = list(self.dataset_map_dic.keys())
        if dataset_name not in name_list:
            raise ValueError("dataset_name should be one of these {}".format(name_list))
        self.dataset_name = dataset_name
    
    def __call__(self,**kwargs):
        return self.dataset_map_dic[self.dataset_name](**kwargs)

class Twitter(Dataset):
    def __init__(self,train=True,**kwargs):
        self.train = train
        all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        all_negative_tweets = twitter_samples.strings('negative_tweets.json')
        random.shuffle(all_positive_tweets)
        random.shuffle(all_negative_tweets)
        pos_train_x = all_positive_tweets[:4000]
        pos_test_x = all_positive_tweets[4000:]
        neg_train_x = all_negative_tweets[:4000]
        neg_test_x = all_negative_tweets[4000:]
        
        self.train_x = pos_train_x + neg_train_x
        self.test_x = pos_test_x + neg_test_x
        
        self.train_y = np.append(np.ones((len(pos_train_x), 1)), np.zeros((len(neg_train_x), 1)), axis=0)
        self.test_y = np.append(np.ones((len(pos_test_x), 1)), np.zeros((len(neg_test_x), 1)), axis=0)
        
        self.preprocessor =  Preprocessor(name="LR_twitter")(train_x=self.train_x,train_y=self.train_y)
        
    def __getitem__(self,index):
        if self.train:
            return self.preprocessor(self.train_x[index],self.train_y[index])
        else:
            return self.preprocessor(self.test_x[index],self.test_y[index])
        
    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)
        
    
    def __str__(self):
        return "Twitter Dataset Loader"
    
