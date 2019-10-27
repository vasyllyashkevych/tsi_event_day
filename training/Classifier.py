#from PIL import Image
import os
import glob
import cv2 as cv2
import pprint
import numpy as np
import logging
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.python.framework import graph_util

print("\nSet up initial settings ... ")
configuration = {
    "path": "ds-vasyl-lyashkevych//",
    "height": 224,
    "width": 224,
    "channel": 3,
    "ratio": 0.8,
    "batch_size": 2,
    "num_epochs": 10,
    "pb_file_path": "parking_vgg.pb"
    }
pprint.pprint(configuration)
print("\n")


def read_img_hdd(path):
    clss = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    print("Classes", clss)
    imgs = []
    labels = []
    for idx, folder in enumerate(clss):
        print("Class: ", str(idx), " is being read ...")
        for q, im in enumerate(glob.glob(folder + '/*.png')):
            print(str(q) + ': The image: %s' % (im) + " is being read ...")
            img = cv2.resize(cv2.imread(im), (configuration['height'], configuration['width']), interpolation=cv2.INTER_AREA)
            imgs.append(img)
            labels.append(idx)
    print("Data-set is read ! \n")
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def build_network():
    x = tf.placeholder(tf.float32, shape=[None, configuration['height'], configuration['width'], configuration['channel']], name='input')
    y = tf.placeholder(tf.int64, shape=[None, 2], name='labels_placeholder')

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def fc(input, w, b):
        return tf.matmul(input, w) + b

    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 3, 64])
        biases = bias_variable([64])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)

    pool1 = pool_max(output_conv1_2)

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)

    pool2 = pool_max(output_conv2_2)

    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)

    with tf.name_scope('conv3_3') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)

    pool3 = pool_max(output_conv3_3)

    # conv4
    with tf.name_scope('conv4_1') as scope:
        kernel = weight_variable([3, 3, 256, 512])
        biases = bias_variable([512])
        output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)

    with tf.name_scope('conv4_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)

    with tf.name_scope('conv4_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)

    pool4 = pool_max(output_conv4_3)

    # conv5
    with tf.name_scope('conv5_1') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)

    with tf.name_scope('conv5_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)

    with tf.name_scope('conv5_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)

    pool5 = pool_max(output_conv5_3)

    #fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        kernel = weight_variable([shape, 4096])
        biases = bias_variable([4096])
        pool5_flat = tf.reshape(pool5, [-1, shape])
        output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

    #fc7
    with tf.name_scope('fc7') as scope:
        kernel = weight_variable([4096, 4096])
        biases = bias_variable([4096])
        output_fc7 = tf.nn.relu(fc(output_fc6, kernel, biases), name=scope)

    #fc8
    with tf.name_scope('fc8') as scope:
        kernel = weight_variable([4096, 2])
        biases = bias_variable([2])
        output_fc8 = tf.nn.relu(fc(output_fc7, kernel, biases), name=scope)

    finaloutput = tf.nn.softmax(output_fc8, name="softmax")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=y))
    optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    prediction_labels = tf.argmax(finaloutput, axis=1, name="output")
    read_labels = y

    correct_prediction = tf.equal(prediction_labels, read_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        x=x,
        y=y,
        optimize=optimize,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
    )


def train_network(graph, x_train, y_train, x_val, y_val, batch_size, num_epochs, pb_file_path):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        epoch_delta = 2
        for epoch_index in range(num_epochs):
            for i in range(12):
                sess.run([graph['optimize']], feed_dict={
                    graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                    graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                })
            if epoch_index % epoch_delta == 0:
                total_batches_in_train_set = 0
                total_correct_times_in_train_set = 0
                total_cost_in_train_set = 0.
                for i in range(12):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    })
                    total_batches_in_train_set += 1
                    total_correct_times_in_train_set += return_correct_times_in_batch
                    total_cost_in_train_set += (mean_cost_in_batch * batch_size)


                total_batches_in_test_set = 0
                total_correct_times_in_test_set = 0
                total_cost_in_test_set = 0.
                for i in range(3):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    })
                    total_batches_in_test_set += 1
                    total_correct_times_in_test_set += return_correct_times_in_batch
                    total_cost_in_test_set += (mean_cost_in_batch * batch_size)

                acy_on_test  = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                print('Epoch - {:2d}, acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}'.format(epoch_index, acy_on_test*100.0,total_correct_times_in_test_set,
                                                                                                                                                   total_batches_in_test_set * batch_size,
                                                                                                                                                   total_cost_in_test_set,
                                                                                                                                                   acy_on_train * 100.0,
                                                                                                                                                   total_correct_times_in_train_set,
                                                                                                                                                   total_batches_in_train_set * batch_size,
                                                                                                                                                   total_cost_in_train_set))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


def main():
    data, label = read_img_hdd(configuration['path'])

    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    s = np.int(num_example * configuration['ratio'])
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    g = build_network()
    train_network(g, x_train, y_train, x_val, y_val, configuration['batch_size'], configuration['num_epochs'], configuration['pb_file_path'])
    print("done")

main()





