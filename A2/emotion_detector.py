import argparse
import pickle

import A1.lab2_landmarks as l2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# tf.disable_v2_behavior()


def transform_images_to_features_data(csv_file_path, smiling_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, smiling_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


def split_train_test_data(x, y, testsize):
    return train_test_split(x, y, test_size=testsize)


def allocate_weights_and_biases(stddev, neurons_layer1, neurons_layer2):

    X = tf.placeholder("float", [None, 68, 2])
    Y = tf.placeholder("float", [None, 2])  # 2 output classes

    images_flat = tf.contrib.layers.flatten(X)

    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, neurons_layer1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([neurons_layer2, 2], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([neurons_layer1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([neurons_layer2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }

    return weights, biases, X, Y, images_flat


def multilayer_perceptron(stddev, neurons_layer1, neurons_layer2):

    weights, biases, X, Y, images_flat = allocate_weights_and_biases(stddev, neurons_layer1, neurons_layer2)
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.math.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.math.sigmoid(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer, X, Y


def loss_and_optimiser(learning_rate, training_epochs, display_accuracy_step, logits, training_images, training_labels,
                       test_images, test_labels
                       ):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(training_epochs):
            _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,
                                                               Y: training_labels})
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))

            if epoch % display_accuracy_step == 0:
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))

        print("Optimization Finished!")
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Test Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))




if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='ML model trained on images to detect emotion from input images.')
    arg_parser.add_argument('-i', '--img-dir',
                            help='the path to the directory where the images are',
                            required=True)
    arg_parser.add_argument('-l', '--labels-file',
                            help='the path to the csv labels file',
                            required=True)
    arg_parser.add_argument('-s', '--landmarks-file',
                            help='the path to the face landmarks file',
                            required=True)
    arg_parser.add_argument('-ei', '--emotion-index',
                            help='the index of the smiling column in the labels.csv file',
                            required=True,
                            type=int)
    arg_parser.add_argument('-pd', '--preprocessed-data-file',
                            help='the path to the preprocessed image data file',
                            required=True)

    args = vars(arg_parser.parse_args())
    image_directory = args['img_dir']
    labels_file = args['labels_file']
    landmarks_file = args['landmarks_file']
    emotion_index = args['emotion_index']
    preprocessed_data_file = args['preprocessed_data_file']
    print("Building emotion classification model from images in {},"
          " using labels file at {} and face landmarks file at {}. Index of smiling field in the CSV file is"
          " {}. The image features are written to a pickle file {}".format(image_directory,labels_file, landmarks_file,
                                                                           emotion_index,
                                                                           preprocessed_data_file))

    # X, Y = transform_images_to_features_data(labels_file, emotion_index, image_directory, landmarks_file)
    # x_y = list(zip(X, Y))

    # pickled = (X, Y)
    filename = preprocessed_data_file

    # with open(filename, 'wb') as f:
    #     pickle.dump(pickled, f)

    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)

    