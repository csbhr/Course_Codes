import tensorflow as tf


def train(parmeter_size, sample_train, sample_verify, alpha, times):
    # the placeholder of samples and lables
    x = tf.placeholder(tf.float32, [None, parmeter_size[0]])
    y = tf.placeholder(tf.float32, [None, parmeter_size[2]])

    # the weight and bais of hidden layer
    w_h = tf.Variable(tf.random_normal([parmeter_size[0], parmeter_size[1]]))
    b_h = tf.Variable(tf.random_normal([1, parmeter_size[1]]))

    # the hidden layer
    value_hidden = tf.nn.sigmoid(tf.matmul(x, w_h) + b_h)

    # the weight and bais of output layer
    w_o = tf.Variable(tf.random_normal([parmeter_size[1], parmeter_size[2]]))
    b_o = tf.Variable(tf.random_normal([1, parmeter_size[2]]))

    # the output layer
    h = tf.nn.sigmoid(tf.matmul(value_hidden, w_o) + b_o)

    # cost function
    diff = tf.subtract(y, h)
    const_temp1 = tf.constant(2.0)
    cost = tf.divide(tf.reduce_sum(tf.multiply(diff, diff)), const_temp1)

    # update with GD
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    # initial
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # training
    for i in range(times):
        print("training:", i)
        sess.run(train_step, feed_dict={x: sample_train[0], y: sample_train[1]})

    # get parameters
    w_h_r = sess.run(w_h)
    b_h_r = sess.run(b_h)
    w_o_r = sess.run(w_o)
    b_o_r = sess.run(b_o)

    # the accuracy of verifying
    diff_abs = tf.abs(diff)
    const_temp2 = tf.constant(0.5)
    diff_div = tf.subtract(diff_abs, const_temp2)
    const_temp3 = tf.constant(-1.0)
    diff_div = tf.multiply(diff_div, const_temp3)
    diff_div = tf.nn.relu(diff_div)
    diff_div = tf.cast(diff_div, "bool")
    diff_div = tf.cast(diff_div, "float")
    accuracy = tf.reduce_mean(diff_div)

    # verifying
    accuracy_r = sess.run(accuracy, feed_dict={x: sample_verify[0], y: sample_verify[1]})

    return w_h_r, b_h_r, w_o_r, b_o_r, accuracy_r
