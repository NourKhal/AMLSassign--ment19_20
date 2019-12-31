import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import A1.lab2_landmarks as l2
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf1
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def transform_images_to_features_data(csv_file_path, smiling_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, smiling_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y

def split_train_test_data(x, y, testsize):
    return train_test_split(x, y, test_size=testsize)


def allocate_weights_and_biases(n_classes):
    X = tf.placeholder("float", [None, 68, 2]) # 68 coordinates of X and Y pairs as the input
    Y = tf.placeholder("float", [None, 5])

    weights = {
        'wc1': tf.get_variable('w0', shape=(3,3,3,3), initializer=tf1.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable('W1', shape=(3,3,96,64), initializer=tf1.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable('W2', shape=(3,3,192,128), initializer=tf1.contrib.layers.xavier_initializer()),
        'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf1.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf1.contrib.layers.xavier_initializer()),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf1.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf1.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf1.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B3', shape=(128), initializer=tf1.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B4', shape=(5), initializer=tf1.contrib.layers.xavier_initializer()),
    }

    return weights, biases, X, Y


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='ML model trained on images to classify face shapes from input images.')
    arg_parser.add_argument('-i', '--img-dir',
                            help='the path to the directory where the images are',
                            required=True)
    arg_parser.add_argument('-l', '--labels-file',
                            help='the path to the csv labels file',
                            required=True)
    arg_parser.add_argument('-s', '--landmarks-file',
                            help='the path to the face landmarks file',
                            required=True)
    arg_parser.add_argument('-fsi', '--face-shape-index',
                            help='the index of the face shape column in the labels.csv file',
                            required=True,
                            type=int)
    arg_parser.add_argument('-pd', '--preprocessed-data-file',
                            help='the path to the preprocessed image data file',
                            required=True)

    args = vars(arg_parser.parse_args())
    image_directory = args['img_dir']
    labels_file = args['labels_file']
    landmarks_file = args['landmarks_file']
    face_shape_index = args['face_shape_index']
    preprocessed_data_file = args['preprocessed_data_file']
    print("Building face shape classification model from images in {},"
          " using labels file at {} and face landmarks file at {}. Index of face shape field in the CSV file is"
          " {}. The image features are written to a pickle file {}".format(image_directory,labels_file, landmarks_file,
                                                                           face_shape_index,
                                                                           preprocessed_data_file))

    filename = preprocessed_data_file

    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)

    X_train, X_test, Y_train, Y_test = split_train_test_data(X, Y, testsize=0.3)
    X_test, X_val, Y_test, Y_val = split_train_test_data(X_test, Y_test, testsize=0.5)

    pickled_train = (X_train, Y_train)
    pickled_val = (X_val, Y_val)
    pickled_test = (X_test, Y_test)

    filename_train = 'face_shape_pickled_train'
    with open(filename_train, 'rb') as f:
        X_train, Y_train = pickle.load(f)

    filename_val = 'face_shape_pickled_val'
    with open(filename_val, 'rb') as f:
        X_val, Y_val = pickle.load(f)

    filename_test = 'face_shape_pickled_test'
    with open(filename_test, 'rb') as f:
        X_test, Y_test = pickle.load(f)

