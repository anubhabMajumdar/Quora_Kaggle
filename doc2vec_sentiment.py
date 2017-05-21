import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
import nltk.data
import logging
from gensim.models import word2vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from random import shuffle
from sklearn import svm
import nn_2 as nn
import tensorflow as tf
import random

num_features = 400    # Word vector dimensionality 
epochCount = 20
batch_size = 100

class LabeledLineSentence(object):
	def __init__(self, sources):
		self.sources = sources
		
		flipped = {}
		
		# make sure that keys are unique
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')
	
	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
	
	def to_array(self):
		self.sentences = []
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
		return self.sentences
	
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences
		# return np.random.permutation(self.sentences)
# ****************************************************************



# ****************************************************************

# # Get sentences
# print "Get sentences"
# sources = {'test.txt':'TEST', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}
# sentences = LabeledLineSentence(sources)      

# # Get gensim Doc2vec model
# print "Get model"
# model = Doc2Vec(min_count=1, window=10, size=num_features, sample=1e-4, negative=5, workers=4)
# model.build_vocab(sentences.to_array())

# # Train Doc2vec model
# print "Train model"
# for epoch in range(epochCount):
# 	print "Epoch = ", epoch
# 	model.train(sentences.sentences_perm())

# # save model
# print "Saving model"
# fileName = "doc2vec_model_"+str(num_features)+"features_"+str(epochCount)+"_epoch"
# model.save(fileName)

print "Loading model"
fileName = "doc2vec_model_"+str(num_features)+"features_"+str(epochCount)+"_epoch"
model = Doc2Vec.load(fileName)

# get training data and labels
print "Getting training data and label"
train_arrays = np.zeros((25000, num_features))
train_labels = np.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0	

labels_matrix = []
for i in train_labels:
	if int(i) == 0:
		labels_matrix.append([1,0])
	else:
		labels_matrix.append([0,1])	

# get testing data
print "Getting testing data"
test_arrays = np.zeros((25000, num_features))

for i in range(25000):
    prefix_test = 'TEST_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test]

# test

# ****************************************************************
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )

# forest = RandomForestClassifier( n_estimators = 100 )
# forest = svm.LinearSVC()
forest = forest = KNeighborsClassifier(n_neighbors=7)

print "Training"
forest = forest.fit(train_arrays, train_labels)

print "Predicting"
result = forest.predict(test_arrays)


# ****************************************************************
# sess = tf.InteractiveSession()
# HL_SIZE = 50
# x, y, train_step, correct_prediction, accuracy, predicted_class = nn.network(sess, num_features, HL_SIZE)
# sess.run(tf.global_variables_initializer())

# print "Training"

# for j in range(3000):

# 	random_index = random.sample(range(0, 25000), batch_size)

# 	batch_x = [train_arrays[i] for i in random_index]
# 	batch_y = [labels_matrix[i] for i in random_index]
	
# 	train_step.run(feed_dict={x: batch_x, y: batch_y})

# 	if j%100 == 0:
# 		train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
# 		print("step %d, training accuracy %g"%(j, train_accuracy))

# print "Predicting"
# test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )

# result = []	
# for i in xrange(0,len(test_arrays), 100):
# 	pl = predicted_class.eval(feed_dict={x:test_arrays[i:i+100]})
# 	result.extend(pl)
# 	print i, " done"


# ****************************************************************

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# fileName = "Doc2vec_model_"+str(num_features)+"features_"+str(epochCount)+"_epoch_NN_"+str(HL_SIZE)+"HL_2HiddenLayer.csv"
# fileName = "Doc2vec_model_"+str(num_features)+"features_"+str(epochCount)+"_epoch_SVM.csv"
fileName = "Doc2vec_model_"+str(num_features)+"features_"+str(epochCount)+"_KNN.csv"
output.to_csv( fileName, index=False, quoting=3 )


