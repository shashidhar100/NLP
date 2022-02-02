import pandas as pd
import os

#library that contains punctuation
import string

import nltk
from nltk.tokenize import SpaceTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords') # downloading stop words.
nltk.download("wordnet") # downloading wordnet for lemmatixation.

STOPWORDS = nltk.corpus.stopwords.words('english') #stop words present in the library.


def remove_punctuation(text):
    """
    Removes the punctuation from the given sentence.
    
    Inputs:
        text: (string) sentence. 
    Outputs: 
        punctuationfree: (string) with punctaution free, 
                        punctuations are given by string.punctuation library.
    """
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree



def lowering_text(text):
    """
    Converts all the characters in the text to lowercase.

    Inputs:
        text: (string).
    Outputs: 
        text: (string) all the characters are in lowercase.
    """
    return text.lower()


def tokenization(text):
    """
    White space tokenization of the text.

    Inputs: 
        text: (string).
    Outputs:
        tokens: (List) list of all the words from the given text which are 
                white space tokenized.
    """
    tk = SpaceTokenizer()
    tokens = tk.tokenize(text)
    return tokens


def remove_stopwords(words_list):
    """
    Removes the stop words from the list of words (words_list).

    Inputs:
        words_list: (List or Tuple) of words.
    Outputs:
        no_stop_words_list: (List) of words without stop words.

    """
    no_stop_words_list = [i for i in words_list if i not in STOPWORDS]
    return no_stop_words_list


def stemming(words_list):
    """
    Performs the stemming on each word from the given word list (words_list).

    Inputs:
        words_list: (List or Tuple) of words.
    Outputs:
        stemmed_words_list: (List) of words which are conveterd to their base (morpheme) form.
    """
    porter_stemmer = PorterStemmer() #creating the instance of PorterStemmer().
    stemmed_words_list = [porter_stemmer.stem(word) for word in words_list]
    return stemmed_words_list


def lemmatizer(words_list):
    """
    Performs the lemmatization on ecah word from the given word list (words_list).

    Inputs: 
        words_list: (List) of words.
    Outputs:
        lemmatized_words_list: (List) of words which are converted to their base (morpheme) form.
    """
    wordnet_lemmatizer = WordNetLemmatizer() #creating the instance of WordNetLemmatizer().
    lemmatized_words_list = [wordnet_lemmatizer.lemmatize(word) for word in words_list]
    return lemmatized_words_list













