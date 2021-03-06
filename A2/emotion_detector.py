import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split

import A1.lab2_landmarks as l2


# extract images feature i.e. facelandmarks
def transform_images_to_features_data(csv_file_path, smiling_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, smiling_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


def split_train_test_data(x, y, testsize):
    return train_test_split(x, y, test_size=testsize)


def allocate_weights_and_biases(stddev, neurons_layer1, neurons_layer2, neurons_layer3):

    ## Set the input data as placeholders to be used later when building the computational graph
    # i.e create a place in memory to store the value later on
    X = tf.placeholder("float", [None, 68, 2], name='X') # 68 coordinates of X and Y pairs as the input
    Y = tf.placeholder("float", [None, 2], name='Y') # 2 ouputs since we have binary classification (0 or 1)

    # Reshape the image features matrix into one vector
    images_flat = tf1.contrib.layers.flatten(X)

    ## Initialise and set the weights of the different layers of the MLP
    # stddev is the standard deviation of the normal distribution
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, neurons_layer1], stddev=stddev)), # this will return a
        # tensor of the specified shape '[68 * 2, neurons_layer1]' filled with random normal values. Similarly, for
        # 'hidden_layer2' and 'out'
        'hidden_layer2': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2], stddev=stddev)),
        'hidden_layer3': tf.Variable(tf.random_normal([neurons_layer2, neurons_layer3], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([neurons_layer3, 2], stddev=stddev))
    }

    ## Initialise and set the biases of the different layers of the MLP
    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([neurons_layer1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([neurons_layer2], stddev=stddev)),
        'bias_layer3': tf.Variable(tf.random_normal([neurons_layer3], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }

    return weights, biases, X, Y, images_flat


def build_multilayer_perceptron(stddev, neurons_layer1, neurons_layer2, neurons_layer3):

    weights, biases, X, Y, images_flat = allocate_weights_and_biases(stddev, neurons_layer1, neurons_layer2, neurons_layer3)
# Return a tensor of the sum of weighted matrix and the biases of the first layer
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.math.sigmoid(layer_1)

    # Return a tensor of the sum of weighted matrix and the biases of the second layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.math.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['hidden_layer3']), biases['bias_layer3'])
    layer_3 = tf.math.sigmoid(layer_3)

    # Return a tensor of the sum of weighted matrix and the biases of the outer layer (output)
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer, X, Y


def loss_calculator_and_optimiser(learning_rate, training_epochs, display_accuracy_step, logits, training_images, training_labels,
                                  test_images, test_labels):
    """Softmax takes a vector of real-valued arguments and transforms it to a vector whose elements fall in the range (0, 1) and sum to 1
    i.e measures the probability error in discrete classification tasks in which the classes are mutually exclusive.
    this function will return a tensor that contains the softmax cross entropy loss"""

    ## The tf.reduce_mean(tensor) will return a reduce tensor with mean of each element in the tensor
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #backpropagation
    train_op = optimizer.minimize(loss_op) #

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        sess.run(init)
        summary_writer = tf.summary.FileWriter('./Output1', sess.graph)
        saver = tf.train.Saver()
        for epoch in range(training_epochs):
            _, train_cost = sess.run([train_op, loss_op], feed_dict={X: training_images, Y: training_labels})
            _, val_cost = sess.run([train_op, loss_op], feed_dict={X: test_images, Y: test_labels})

            train_loss.append(train_cost)
            test_loss.append(val_cost)

            print("Epoch:", '%04d' % (epoch + 1), "training cost={:.9f}".format(train_cost))
            print("Epoch:", '%04d' % (epoch + 1), "validation cost={:.9f}".format(val_cost))

            if epoch % display_accuracy_step == 0:
                pred = tf.nn.softmax(logits)
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                current_accuracy = accuracy.eval({X: training_images, Y: training_labels})
                validation_accuracy = accuracy.eval({X: test_images, Y: test_labels})
                train_accuracy.append(current_accuracy)
                test_accuracy.append(validation_accuracy)

                print("Training Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))
                print("Test Accuracy: {:.3f}".format(accuracy.eval({X: test_images, Y: test_labels})))

        print("Optimization Task Completed!")
        pred = tf.nn.softmax(logits, name='pred')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        saver.save(sess, 'emotion-detector-model1')

        print("Test Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))
        plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
        plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
        plt.title('Training and Test loss')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()

        plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Training Accuracy')
        plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()
    summary_writer.close()

def restore_model(X_test, Y_test):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('emotion-detector-model1.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    pred = graph.get_tensor_by_name("accuracy:0")
    x= graph.get_tensor_by_name("X:0")
    y_true = graph.get_tensor_by_name("Y:0")
    X_test = (X_test - np.min(X_test))/ (np.max(X_test) - np.min(X_test))
    feed_dict_testing = {x: X_test, y_true: Y_test}
    result=sess.run(pred, feed_dict=feed_dict_testing)
    print(result)



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

    filename = preprocessed_data_file
    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)

    filename_train = 'pickled_train'
    with open(filename_train, 'rb') as f:
        X_train, Y_train = pickle.load(f)

    filename_val = 'pickled_val'
    with open(filename_val, 'rb') as f:
        X_val, Y_val = pickle.load(f)

    filename_test = 'pickled_test'
    with open(filename_test, 'rb') as f:
        X_test, Y_test = pickle.load(f)
    #
    logits, X, Y = build_multilayer_perceptron(0.01, 500, 250, 125)
    loss_calculator_and_optimiser(0.0001, 600, 2, logits, X_train, Y_train, X_test, Y_test)

    restore_model(X_test, Y_test)
