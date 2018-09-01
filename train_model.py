import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Create mnist model
def mnist_model():
# Import data
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # Placeholder that will be fed image data.
    x = tf.placeholder(tf.float32, [None, 784])
    # Placeholder that will be fed the correct labels.
    y_ = tf.placeholder(tf.float32, [None, 10])
    # Define weight and bias.
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # Here we define our model which utilizes the softmax regression.
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # Define our loss.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # Define our optimizer.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # Normal TensorFlow - initialize values, create a session and run the model
    model = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(model)
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Save the variables(model) to disk.
        save_path = saver.save(sess, "./mnist_model/model.ckpt")
        print("Model saved in path: %s" % save_path)
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("predict accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

#####

mnist_model()
