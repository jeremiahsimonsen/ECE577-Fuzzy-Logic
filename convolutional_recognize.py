import tensorflow as tf
import input_data
import display

# Import training and testing data sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Helper functions for initializing variables
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
		strides=[1, 2, 2, 1], padding='SAME')

# Build the model

# Inputs and outputs
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# First convolutional layer
# Patch weights and biases
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# Reshape image for convolution
x_image = tf.reshape(x,[-1,28,28,1])

# Convolve and down-sample using max-pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
# Patch weights and biases
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Convolve and down-sample using max-pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# Dropout to avoid overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer of softmax
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# Minimize cross entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# For saving model state for later use
saver = tf.train.Saver()

# Perform the recognition
with tf.Session() as sess:
	# Open and restore model
	saver.restore(sess, "MNIST_data/model.ckpt-19000")
	print("Model restored.")

	# Feed images into neural net and print predictions
	prediction=tf.argmax(y_conv,1)
	best = sess.run([prediction],
		feed_dict={x:mnist.zipcode.images, keep_prob: 1.0})
	print(best)

# Display the images with predictions
display.display_zip(mnist.zipcode.images,best[0])