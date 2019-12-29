import argparse
import pickle

import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import A1.lab2_landmarks as l2


def transform_images_to_features_data(csv_file_path, smiling_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, smiling_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


def split_train_test_data(x, y, testsize):
    return train_test_split(x, y, test_size=testsize)


def allocate_weights_and_biases(stddev, neurons_layer1, neurons_layer2):

    ## Set the input data as placeholders to be used later when building the computational graph
    # i.e create a place in memory to store the value later on
    X = tf.placeholder("float", [None, 68, 2]) # 68 coordinates of X and Y pairs as the input
    Y = tf.placeholder("float", [None, 2]) # 2 ouputs since we have binary classification (0 or 1)

    # Reshape the image matrix features into one vector
    images_flat = tf1.contrib.layers.flatten(X)

    ## Initialise and set the weights of the different layers of the MLP
    # stddev is the standard deviation of the normal distribution
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, neurons_layer1], stddev=stddev)), # this will return a
        # tensor of the specified shape '[68 * 2, neurons_layer1]' filled with random normal values. Similarly, for
        # 'hidden_layer2' and 'out'
        'hidden_layer2': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([neurons_layer2, 2], stddev=stddev))
    }

    ## Initialise and set the biases of the different layers of the MLP
    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([neurons_layer1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([neurons_layer2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }

    return weights, biases, X, Y, images_flat


def build_multilayer_perceptron(stddev, neurons_layer1, neurons_layer2):

    weights, biases, X, Y, images_flat = allocate_weights_and_biases(stddev, neurons_layer1, neurons_layer2)

    # Return a tensor of the sum of weighted matrix and the biases of the first layer
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.math.sigmoid(layer_1)

    # Return a tensor of the sum of weighted matrix and the biases of the second layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.math.sigmoid(layer_2)

    # Return a tensor of the sum of weighted matrix and the biases of the outer layer (output)
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    return out_layer, X, Y


def loss_and_optimiser(learning_rate, training_epochs, display_accuracy_step, logits, training_images, training_labels,
                       test_images, test_labels):
    """Softmax takes a vector of real-valued arguments and transforms it to a vector whose elements fall in the range (0, 1) and sum to 1
    i.e measures the probability error in discrete classification tasks in which the classes are mutually exclusive.
    this function will return a tensor that contains the softmax cross entropy loss"""

    ## The tf.reduce_mean(tensor) will return a reduce tensor with mean of each element in the tensor
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op) #

    init = tf.global_variables_initializer()

    epoch_cost_plot = []
    cost = []
    epoch_validation_plot = []
    validation = []
    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(training_epochs):
            _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,
                                                               Y: training_labels})
            epoch_cost_plot.append(epoch+1)
            cost.append(cost)

            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))

            if epoch % display_accuracy_step == 0:
                pred = tf.nn.softmax(logits)
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                current_accuracy = accuracy.eval({X: training_images, Y: training_labels})
                epoch_validation_plot.append(epoch)
                validation.append( current_accuracy* 100)
                print("Training Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))

        print("Optimization Task Completed!")
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Validation Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))

        # Plot results
        plt.figure(1)
        plt.subplot(211)
        plt.plot(epoch_cost_plot, cost)
        plt.xlabel('Epoch')
        plt.ylabel('Cost/Error')
        plt.axis([0, epoch_cost_plot[-1], 0, 1])

        plt.subplot(212)
        plt.plot(epoch_validation_plot, validation)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.show()


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

    # X_train, X_test, Y_train, Y_test = split_train_test_data(X, Y, testsize=0.3)
    # X_test, X_val, Y_test, Y_val = split_train_test_data(X_test, Y_test, testsize=0.5)

    # pickled_train = (X_train, Y_train)
    # pickled_val = (X_val, Y_val)
    # pickled_test = (X_test, Y_test)

    filename_train = 'pickled_train'
    with open(filename_train, 'rb') as f:
        X_train, Y_train = pickle.load(f)

    filename_val = 'pickled_val'
    with open(filename_val, 'rb') as f:
        X_val, Y_val = pickle.load(f)

    filename_test = 'pickled_test'
    with open(filename_test, 'rb') as f:
        X_test, Y_test = pickle.load(f)

    logits, X, Y = build_multilayer_perceptron(0.01, 100, 100)
    loss_and_optimiser(0.001, 500, 2, logits, X_train, Y_train, X_val, Y_val)

