import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def network(sess, max_features, hl_features):
	x = tf.placeholder(tf.float32, shape=[None, max_features])
	y = tf.placeholder(tf.float32, shape=[None, 2])

	######################################################################################

	w1 = weight_variable([max_features, hl_features])
	b1 = bias_variable([hl_features])

	h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
	# print h1.shape()
	######################################################################################

	w2 = weight_variable([hl_features, hl_features])
	b2 = bias_variable([hl_features])

	# h1_transpose = tf.reshape(h1, [-1, hl_features])
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

	######################################################################################

	w3 = weight_variable([hl_features, 2])
	b3 = bias_variable([2])

	y_nn = tf.matmul(h2, w3) + b3

	######################################################################################

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_nn))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 

	correct_prediction = tf.equal(tf.argmax(y_nn,1), tf.argmax(y,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	predicted_class = tf.argmax(y_nn,1)

	return x, y, train_step, correct_prediction, accuracy, predicted_class