import os
import glob
import cv2 as cv2
import numpy as np
import pprint
import logging
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)
# logging.getLogger('tensorflow').disabled = True
from tensorflow.python.framework import graph_util


print("\nSet up initial settings ... ")
configuration = {
    "path": "ds-vasyl-lyashkevych//",
    "height": 224,
    "width": 224,
    "channel": 3,
    "ratio": 0.8,
    "batch_size": 2,
    "num_epochs": 1000000,
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
    x = tf.compat.v1.placeholder(tf.float32,
                                 shape=[None, configuration['height'],
                                                    configuration['width'],
                                                    configuration['channel']],
                                 name='input')
    y = tf.compat.v1.placeholder(tf.int64, shape=[None, 3], name='labels_placeholder')

    def weight_variable(shape, name="weights"):
        initial = tf.random.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(input):
        return tf.nn.max_pool2d(input,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')

    def fc(input, w, b):
        return tf.matmul(input, w) + b

    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = weight_variable([3, 3, 3, 16])
        biases = bias_variable([16])
        output_conv1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = weight_variable([3, 3, 16, 32])
        biases = bias_variable([32])
        output_conv2 = tf.nn.relu(conv2d(output_conv1, kernel) + biases, name=scope)

    pool2 = pool_max(output_conv2)

    #fc5
    with tf.name_scope('fc5') as scope:
        shape = int(np.prod(pool2.get_shape()[1:]))
        kernel = weight_variable([shape, 1024])
        biases = bias_variable([1024])
        pool5_flat = tf.reshape(pool2, [-1, shape])
        output_fc5 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

    #fc6
    with tf.name_scope('fc6') as scope:
        kernel = weight_variable([1024, 3])
        biases = bias_variable([3])
        output_fc6 = tf.nn.relu(fc(output_fc5, kernel, biases), name=scope)

    finaloutput = tf.nn.softmax(output_fc6, name="softmax")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=y))
    optimize = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

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
        accuracy=accuracy
    )

def get_vector(x):
    if x == 0:
        return [[1, 0, 0]]
    elif x == 1:
        return [[0, 1, 0]]
    elif x == 2:
        return [[0, 0, 1]]


def train_network(graph, x_train, y_train, x_val, y_val, batch_size, num_epochs, pb_file_path):
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        epoch_delta = 2
        for epoch_index in range(num_epochs):
            for i in range(12):
                sess.run([graph['optimize']], feed_dict={
                    graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                    graph['y']: get_vector(y_train[i])
                })
            if epoch_index % epoch_delta == 0:
                total_batches_in_train_set = 0
                total_correct_times_in_train_set = 0
                total_cost_in_train_set = 0.
                for i in range(12):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        graph['y']: get_vector(y_train[i])
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        graph['y']: get_vector(y_train[i])
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
                        graph['y']: get_vector(y_train[i])
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        graph['y']: get_vector(y_train[i])
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
            with tf.compat.v1.gfile.FastGFile(pb_file_path, mode='wb') as f:
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




