import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import spatial
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from math import*
from sklearn import tree

max_features = 3000

def euclidean_distance(x,y):
 	return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( set(meaningful_words) ))   

######################################################################################

def bow(clean_train_reviews):
	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 5000) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(clean_train_reviews)

	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	train_data_features = train_data_features.toarray()

	return train_data_features, vectorizer

def preprocessBOW(array_of_bow):
	for i in array_of_bow:
		for j in i:
			if j>=1:
				j = 1
	return array_of_bow			

def distance_measure(q1, q2):
	l = len(q1)
	d = 0.0
	for i in range(l):
		if q1[i] != q2[i]:
			d += 1

	result = d/float(l)
	return result		

######################################################################################

def getFeaturesAndLabels(fileName):
	trainData = pd.read_csv(fileName)
	f = trainData.replace(np.nan, "", regex=True)

	results = []
	labels = []
	for i in range(len(f)):
		
		q1 = f['question1'].loc[i]
		q2 = f['question2'].loc[i]
		
		x = []
		x.append(review_to_words(q1))
		x.append(review_to_words(q2))

		try:
			b, _ = bow(x)
			b = preprocessBOW(b)
			result = 1 - distance_measure(b[0], b[1])
		except:
			result = 0
		
		results.append(result)

		labels.append(f['is_duplicate'].loc[i])

		print i, " done"
		# break

	return results, labels

######################################################################################

def getFeatures(fileName):
	trainData = pd.read_csv(fileName)
	f = trainData.replace(np.nan, "", regex=True)

	results = []
	for i in range(len(f)):
		
		q1 = f['question1'].loc[i]
		q2 = f['question2'].loc[i]
		
		x = []
		x.append(review_to_words(q1))
		x.append(review_to_words(q2))

		try:
			b, _ = bow(x)
			b = preprocessBOW(b)
			result = 1 - distance_measure(b[0], b[1])
		except:
			result = 0
		
		results.append(result)

		print i, " done"
		# break

	return results

######################################################################################

# features, labels = getFeaturesAndLabels("train.csv")

######################################################################################

# print "Writing predictions"

# # Copy the results to a pandas dataframe with an "id" column and
# # a "sentiment" column
# # output = pd.DataFrame( data={"Q1":f['question1'], "Q2":f['question2'], "Similarity Measure":results, "Result":labels} )
# output = pd.DataFrame( data={"Similarity Measure":features, "Result":labels} )

# # Use pandas to write the comma-separated output file
# output.to_csv( "Custom_Similarity_Measure.csv", index=False, quoting=3 )

######################################################################################
print "Read train data features"

fp = pd.read_csv("Custom_Similarity_Measure.csv")

features = fp['Similarity Measure']
labels = fp['Result']

######################################################################################

# test_features = getFeatures('test.csv')

######################################################################################

# output = pd.DataFrame( data={"test_features": test_features})
# output.to_csv( "Custom_similarity_measure_testData.csv", index=False, quoting=3 )

######################################################################################
print "Fit train data features"

X = features[:, None]

# clf = tree.DecisionTreeClassifier()
# clf = svm.LinearSVC()
clf = svm.SVC(kernel='rbf', degree=3)
clf = clf.fit(X, labels)

######################################################################################
print "Read test data features"

fp = pd.read_csv("Custom_similarity_measure_testData.csv")

test_features = fp['test_features']

######################################################################################
print "Predict test data features"

Y = test_features[:, None]
test_labels = clf.predict(Y)

######################################################################################
print "Write predictions"

test_f = pd.read_csv('test.csv')
output = pd.DataFrame( data={"test_id":test_f['test_id'], "is_duplicate":test_labels} )
output.to_csv( "BOW_Custom_Similarity_Measure_Predictions.csv", index=False, header=True, columns=["test_id", "is_duplicate"])



