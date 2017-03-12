from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random


epochs = 200
step = 50

# Training dataset
tr_X = numpy.asarray([4.98,9.14,4.03,2.94,5.33,5.21,12.43,19.15,29.93,17.10,20.45,13.27,15.71,8.26,10.26,8.47,6.58,14.67,11.69,11.28,21.02,13.83,18.72,19.88,16.30,16.51,14.81,17.28,12.80,11.98,22.60,13.04,27.71,18.35,20.34,9.68,11.41,8.77,10.13,4.32,1.98,4.84,5.81,7.44,9.55,10.21,14.15,18.80,30.81,16.20,13.45,9.43,5.28,8.43,14.80,4.81,5.77,3.95,6.86] )
tr_Y = numpy.asarray([24.00,21.60,34.70,33.40,36.20,28.70,22.90,27.10,16.50,18.90,15.00,18.90,21.70,20.40,18.20,19.90,23.10,17.50,20.20,18.20,13.60,19.60,15.20,14.50,15.60,13.90,16.60,14.80,18.40,21.00,12.70,14.50,13.20,13.10,13.50,18.90,20.00,21.00,24.70,30.80,34.90,26.60,25.30,24.70,21.20,19.30,20.00,16.60,14.40,19.40, 19.70, 20.50,25.00,23.40,18.90,35.40,24.70,31.60,23.30])
n_samples = tr_X.shape[0]

# Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Linear model
predict = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(predict-Y, 2))/(2*n_samples)
# Gradient descent
train_optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)


    for epoch in range(epochs):
        for (x, y) in zip(tr_X, tr_Y):
            sess.run(train_optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % step == 0:
            c = sess.run(cost, feed_dict={X: tr_X, Y:tr_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Completed!")
    training_cost = sess.run(cost, feed_dict={X: tr_X, Y: tr_Y})
    print("Train cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(tr_X, tr_Y, 'ro', label='Actual data')
    plt.plot(tr_X, sess.run(W) * tr_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing
    test_X = numpy.asarray([9.22,13.15,14.44 ,6.73 ,9.50 ,8.05 ,4.67,10.24 ,8.10,13.09 ,8.79 ,6.72 ,9.88 ,5.52 ,7.54 ,6.78 ,8.94,11.97,10.27,12.34 ,9.10 ,5.29 ,7.22 ,6.72 ,7.51 ,9.62 ,6.53,12.86 ,8.44 ,5.50 ,5.70 ,8.81 ,8.20 ,8.16 ,6.21,10.59 ,6.65,11.34 ,4.21 ,3.57 ,6.19 ,9.42 ,7.67,10.63,13.44,12.33,16.47,18.66,14.09,12.27,15.55,13.00,10.16,16.21,17.09,10.45,15.76,12.04,10.30,15.37,13.61,14.37,14.27,17.93,25.41,17.58,14.81,27.26,17.19,15.39,18.34,12.60,12.26,11.12,15.03,17.31,16.96,16.90,14.59,21.32,18.46,24.16,34.41,26.82,26.42,29.29,27.80,16.65,29.53,28.32,21.45])
    test_Y = numpy.asarray([19.60,18.70,16.00,22.20,25.00,33.00,23.50,19.40,22.00,17.40,20.90,24.20,21.70,22.80,23.40,24.10,21.40,20.00,20.80,21.20,20.30,28.00,23.90,24.80,22.90,23.90,26.60,22.50,22.20,23.60,28.70,22.60,22.00,22.90,25.00,20.60,28.40,21.40,38.70,43.80,33.20,27.50,26.50,18.60,19.30,20.10,19.50,19.50,20.40,19.80,19.40,21.70,22.80,18.80,18.70,18.50,18.30,21.20,19.20,20.40,19.30,22.00,20.30,20.50,17.30,18.80,21.40,15.70,16.20,18.00,14.30,19.20,19.60,23.00,18.40,15.60,18.10,17.40,17.10,13.30,17.80,14.00,14.40,13.40,15.60,11.80,13.80,15.60,14.60,17.80,15.40])
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(predict - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})
    print("Test cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(tr_X, sess.run(W) * tr_X + sess.run(b), label='Fitted line')
    plt.legend()
plt.show()