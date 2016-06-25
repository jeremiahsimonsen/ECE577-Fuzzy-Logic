import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

# Import training and testing data sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Build the model

# Input images - 28x28 image flattened to 784 array
x = tf.placeholder(tf.float32, [None, 784])

# Weights and biases for the regular neural net
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Softmax regression for output probability distribution
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

# Train to minimize cross entropy between output and correct probability distributions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Start session and initialize variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train the neural net for 1000 steps
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# Measure accuracy using test images
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Print the prediction
prediction=tf.argmax(y,1)
best = sess.run([prediction],feed_dict={x:mnist.test.images})
print(best)

# Display the first test image for reference
tmp = mnist.test.images[0]
tmp = tmp.reshape((28,28))

plt.imshow(tmp, cmap = cm.Greys)
plt.show()