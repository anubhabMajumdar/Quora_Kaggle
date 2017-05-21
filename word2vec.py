import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
import nltk.data
import logging
from gensim.models import word2vec
import numpy as np
from sklearn import svm
# import nn_2 as nn
import tensorflow as tf
import random
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
