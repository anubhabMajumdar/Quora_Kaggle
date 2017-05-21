import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
import nltk.data
import logging
from gensim.models import word2vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import nn_2 as nn
import tensorflow as tf
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# nltk.download()   
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " \
"and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size )

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

# ****************************************************************

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)


num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


# print "Training model..."
# model = word2vec.Word2Vec(sentences, workers=num_workers, \
#             size=num_features, min_count = min_word_count, \
#             window = context, sample = downsampling)

# # If you don't plan to train the model any further, calling 
# # init_sims will make the model much more memory-efficient.
# model.init_sims(replace=True)

# # It can be helpful to create a meaningful model name and 
# # save the model for later use. You can load it later using Word2Vec.load()
# # model_name = "300features_40minwords_10context"
# model_name = "1000features_40minwords_10context"
# model.save(model_name)

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

model = word2vec.Word2Vec.load("300features_40minwords_10context")

_, num_features = model.wv.syn0.shape

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

# Fit a random forest to the training data, using 100 trees
# forest = RandomForestClassifier( n_estimators = 100 )

# forest = svm.LinearSVC()
# classifier = clf.fit(train_data_features, df['sentiment'])

# forest = KNeighborsClassifier(n_neighbors=3)

# ****************************************************************
epochCount = 20
batch_size = 100

labels_matrix = []
for i in train["sentiment"]:
    if int(i) == 0:
        labels_matrix.append([1,0])
    else:
        labels_matrix.append([0,1]) 

sess = tf.InteractiveSession()
HL_SIZE = 50
x, y, train_step, correct_prediction, accuracy, predicted_class = nn.network(sess, num_features, HL_SIZE)
sess.run(tf.global_variables_initializer())

print "Training"

for j in range(3000):

  random_index = random.sample(range(0, 25000), batch_size)

  batch_x = [trainDataVecs[i] for i in random_index]
  batch_y = [labels_matrix[i] for i in random_index]
    
  train_step.run(feed_dict={x: batch_x, y: batch_y})

  if j%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
      print("step %d, training accuracy %g"%(j, train_accuracy))

print "Predicting"
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )

result = []   
for i in xrange(0,len(test), 100):
  pl = predicted_class.eval(feed_dict={x:testDataVecs[i:i+100]})
  result.extend(pl)
  print i, " done"


# ****************************************************************
# print "Fitting a random forest to labeled training data..."
# forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
# result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors_NN.csv", index=False, quoting=3 )

