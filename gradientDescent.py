import tensorflow as tf
import math


def get_min_x_square():
    x = tf.placeholder(tf.float32)
    x0 = tf.Variable([.3], dtype=tf.float32)

    loss = x0 ** 2 - x ** 2

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    x_train = [-2, -1, 0, 1, 2, 3, 4]
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            sess.run(train, {x: x_train})

        x0, curr_loss = sess.run([x0, loss], {x: x_train})
        print("x^2极小值为" + str(x0[0] ** 2) + "，极小值点为" + str(x0[0]))


def get_min_x_vector():
    x = tf.placeholder(tf.float32)
    x0 = tf.Variable([.3], dtype=tf.float32)

    loss = tf.sqrt((x0 - 1) ** 2 + (x0 - 2) ** 2 + (x0 - 3) ** 2) - tf.sqrt((x - 1) ** 2 + (x - 2) ** 2 + (x - 3) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    x_train = [-2, -1, 0, 1, 2, 3, 4]
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            sess.run(train, {x: x_train})

        x0, curr_loss = sess.run([x0, loss], {x: x_train})
        distance = math.sqrt((x0 - 1) ** 2 + (x0 - 2) ** 2 + (x0 - 3) ** 2)
        print("x的值为" + str(x0[0])+"，最短距离为"+str(distance))


if __name__ == '__main__':
    get_min_x_square()
    get_min_x_vector()
