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

def getPredictions(clean_test_q1, clean_test_q2, model):
	print "Getting features of Q1"
	testDataVecs_q1 = getAvgFeatureVecs( clean_test_q1, model, num_features=300 )

	print "Getting features of Q2"
	testDataVecs_q2 = getAvgFeatureVecs( clean_test_q2, model, num_features=300 )

	# ****************************************************************

	print "Get Similarities"

	test_features = []
	for review1, review2 in zip(testDataVecs_q1, testDataVecs_q2):
		try:
			test_features.append(1 - spatial.distance.cosine(review1, review2))
		except:
			test_features.append(0.0)

	######################################################################################

	print "Predict test data features"

	# Y = test_features[:, None]
	Y = np.reshape(features, (len(test_features), 1))

	test_labels = clf.predict(Y)

	######################################################################################

	return test_labels

# ****************************************************************


trainData = pd.read_csv('train.csv')
train = trainData.replace(np.nan, "", regex=True)

# ****************************************************************

# count = 1
# cannot_convert = 0
# sentences = []  # Initialize an empty list of sentences

# for review1, review2 in zip(train["question1"], train["question2"]):
# 	try:
# 		sentences +=  review_to_sentences(review1.decode("utf8"), tokenizer)
# 	except:
# 		cannot_convert += 1
# 	try:
# 		sentences += review_to_sentences(review2.decode("utf8"), tokenizer)
# 	except:
# 		cannot_convert += 1
# 	print "Train Question for Word2Vec ", count, " done"
# 	count += 1
# 	# if (count > 10000):
# 	# 	break


# ****************************************************************

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# ****************************************************************

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
# model_name = "300features_40minwords_10context"
# model.save(model_name)

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

model = word2vec.Word2Vec.load("300features_40minwords_10context")

# ****************************************************************

count = 1
clean_train_q1 = []
clean_train_q2 = []

for review1, review2 in zip(train["question1"], train["question2"]):
	clean_train_q1.append( review_to_wordlist( review1, remove_stopwords=True ))
	clean_train_q2.append( review_to_wordlist( review2, remove_stopwords=True ))
	print "Train Question ", count, " done"
	count += 1

# ****************************************************************

# # Load Google's pre-trained Word2Vec model.
# print "Loading model..."
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# ****************************************************************

print "Getting features of Q1"
trainDataVecs_q1 = getAvgFeatureVecs( clean_train_q1, model, num_features=300 )

print "Getting features of Q2"
trainDataVecs_q2 = getAvgFeatureVecs( clean_train_q2, model, num_features=300 )

# ****************************************************************

print "Get Similarities"

features = []
for review1, review2 in zip(trainDataVecs_q1, trainDataVecs_q2):
	try:
		features.append(1 - spatial.distance.cosine(review1, review2))
	except:
		features.append(0.0)

# ****************************************************************

print "Fit train data features"

X=np.reshape(features, (len(features), 1))

# clf = tree.DecisionTreeClassifier()
clf = svm.LinearSVC()
# clf = svm.SVC(kernel='rbf', degree=3)
clf = clf.fit(X, train['is_duplicate'])

######################################################################################

trainData = []
train = []

clean_train_q1 = []
clean_train_q2 = []

trainDataVecs_q1 = []
trainDataVecs_q2 = []

features = []
X = []

######################################################################################

print "Read test data"

testData = pd.read_csv('test.csv')
test = testData.replace(np.nan, "", regex=True)

# ****************************************************************

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
		test_labels.extend(getPredictions(clean_test_q1, clean_test_q2, model))
		count = 0
		clean_test_q1 = []
		clean_test_q2 = []
		batch_count+=1
		print "Batch ", batch_count, " done"
		
# ****************************************************************

# print "Getting features of Q1"
# testDataVecs_q1 = getAvgFeatureVecs( clean_test_q1, model, num_features=300 )

# print "Getting features of Q2"
# testDataVecs_q2 = getAvgFeatureVecs( clean_test_q2, model, num_features=300 )

# # ****************************************************************

# clean_test_q1 = []
# clean_test_q2 = []

testData = []
test = []

# ****************************************************************
# # ****************************************************************

# print "Get Similarities"

# test_features = []
# for review1, review2 in zip(testDataVecs_q1, testDataVecs_q2):
# 	try:
# 		test_features.append(1 - spatial.distance.cosine(review1, review2))
# 	except:
# 		test_features.append(0.0)

# ######################################################################################

# testDataVecs_q1 = []
# testDataVecs_q2 = []

# ######################################################################################

# print "Predict test data features"

# # Y = test_features[:, None]
# Y = np.reshape(features, (len(test_features), 1))

# test_labels = clf.predict(Y)

######################################################################################

print "Write predictions"

test_f = pd.read_csv('test.csv')
output = pd.DataFrame( data={"test_id":test_f['test_id'], "is_duplicate":test_labels} )
output.to_csv( "Word2vec_Custom_Similarity_Measure_Predictions.csv", index=False, header=True, columns=["test_id", "is_duplicate"])














