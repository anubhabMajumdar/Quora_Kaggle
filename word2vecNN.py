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
import nn_2 as nn
import tensorflow as tf
import random
import gensim
from scipy import spatial

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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
	# index2word_set = set(model.wv.index2word)
	index2word_set = set(model.index2word)
	#
	# Loop over each word in the review and, if it is in the model's
	# vocaublary, add its feature vector to the total
	for word in words:
		if word in index2word_set: 
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])
	# 
	# Divide the result by the number of words to get the average
	if nwords!=0:
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

trainData = pd.read_csv('train.csv')
train = trainData.replace(np.nan, "", regex=True)

# # ****************************************************************

count = 1
clean_train_q1 = []
clean_train_q2 = []

for review1, review2 in zip(train["question1"], train["question2"]):
	clean_train_q1.append( review_to_wordlist( review1, remove_stopwords=False ))
	clean_train_q2.append( review_to_wordlist( review2, remove_stopwords=False ))
	print "Train Question ", count, " done"
	count += 1

# ****************************************************************

num_features = 300    # Word vector dimensionality    

# # ****************************************************************

# Load Google's pre-trained Word2Vec model.
print "Loading model..."
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

# # ****************************************************************

print "Getting features of Q1"
trainDataVecs_q1 = getAvgFeatureVecs( clean_train_q1, model, num_features=300 )

print "Getting features of Q2"
trainDataVecs_q2 = getAvgFeatureVecs( clean_train_q2, model, num_features=300 )

trainDataVecs = []
for i, j in zip(trainDataVecs_q1, trainDataVecs_q2):
	trainDataVecs.append(np.subtract(i,j))

# # ****************************************************************

print "Cleaning up ..."
train = []
trainDataVecs_q1 = []
trainDataVecs_q2 = []
clean_train_q1 = []
clean_train_q2 = []	

# # ****************************************************************

epochCount = 10000
batch_size = 64

labels_matrix = []
for i in trainData["is_duplicate"]:
    if int(i) == 0:
        labels_matrix.append([1,0])
    else:
        labels_matrix.append([0,1]) 

print "Creating Neural Network"
sess = tf.InteractiveSession()
HL_SIZE = 600
x, y, train_step, correct_prediction, accuracy, predicted_class = nn.network(sess, num_features, HL_SIZE)
sess.run(tf.global_variables_initializer())

print "Training"

for j in range(epochCount):

  random_index = random.sample(range(0, len(trainDataVecs)), batch_size)

  batch_x = [trainDataVecs[i] for i in random_index]
  batch_y = [labels_matrix[i] for i in random_index]
    
  train_step.run(feed_dict={x: batch_x, y: batch_y})

  if j%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
      print("step %d, training accuracy %g"%(j, train_accuracy))

# ****************************************************************

print "Cleaning up ..."
trainDataVecs = []

# ######################################################################################

print "Read test data"

testData = pd.read_csv('test.csv')
test = testData.replace(np.nan, "", regex=True)

# # ****************************************************************

count = 0
clean_test_q1 = []
clean_test_q2 = []
test_labels = []
batch = 1000
batch_count = 0

for review1, review2 in zip(test["question1"], test["question2"]):
	clean_test_q1.append( review_to_wordlist( review1, remove_stopwords=True ))
	clean_test_q2.append( review_to_wordlist( review2, remove_stopwords=True ))
	
	count += 1
	
	if count==batch:
		print "Getting features of Q1"
		testDataVecs_q1 = getAvgFeatureVecs( clean_test_q1, model, num_features=300 )

		print "Getting features of Q2"
		testDataVecs_q2 = getAvgFeatureVecs( clean_test_q2, model, num_features=300 )

		testDataVecs = []
		for i, j in zip(testDataVecs_q1, testDataVecs_q2):
			testDataVecs.append(np.subtract(i,j))

		for i in xrange(0,len(testDataVecs), 100):
		  pl = predicted_class.eval(feed_dict={x:testDataVecs[i:i+100]})
		  test_labels.extend(pl)
		  print i, " done"

		clean_test_q1 = []
		clean_test_q2 = []
		testDataVecs_q1 = []
		testDataVecs_q2 = []
		batch_count+=1
		print "Batch ", batch_count, " done"
		
print "Getting features of Q1"
testDataVecs_q1 = getAvgFeatureVecs( clean_test_q1, model, num_features=300 )

print "Getting features of Q2"
testDataVecs_q2 = getAvgFeatureVecs( clean_test_q2, model, num_features=300 )

testDataVecs = []
for i, j in zip(testDataVecs_q1, testDataVecs_q2):
	testDataVecs.append(np.subtract(i,j))

for i in xrange(0,len(testDataVecs), 100):
  pl = predicted_class.eval(feed_dict={x:testDataVecs[i:i+100]})
  test_labels.extend(pl)
  print i, " done"

# ######################################################################################

print "Write predictions"

test_f = pd.read_csv('test.csv')
output = pd.DataFrame( data={"test_id":test_f['test_id'], "is_duplicate":test_labels} )
output.to_csv( "Word2vec_GoogleNews500000_2LayerNN.csv", index=False, header=True, columns=["test_id", "is_duplicate"])





















