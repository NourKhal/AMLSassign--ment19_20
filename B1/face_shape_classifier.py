import argparse
import os
import sys

from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf1
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
from keras.preprocessing import image
from tqdm import tqdm


def load_images(images_dir, face_shape_index, csv_file_path):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open(csv_file_path, 'r')
    lines = labels_file.readlines()
    face_shape_labels = {line.split('\t')[0] : int(line.split('\t')[face_shape_index]) for line in lines[1:]}
    image_dict = {}
    image_pixels = []
    labels = []
    for img_path in image_paths:
        file_name_and_extension = img_path.split('/')[-1]
        file_name_without_extension = file_name_and_extension.split('.')[0]
        image_dict[int(file_name_without_extension)] = img_path
    image_dict = {k: v for k, v in sorted(image_dict.items(), key=lambda item: item[0])}
    for key, img_path in tqdm(image_dict.items()):
        img_num = img_path.split('/')[-1].split('.')[0]
        if img_num not in face_shape_labels.keys():
            # print("\tImage '{}' in directory does not exist in CSV file '{}' - skipping...".format(img_num,
            #                                                                                        csv_file_path))
            continue
        if img_num in img_path:
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=None))
            if img is not None:
                image_pixels.append(img)
                labels.append(face_shape_labels[img_num])

    image_pixels = np.array(image_pixels)
    labels = np.array(labels)
    labels = np.array([labels]).T
    labels = MultiLabelBinarizer().fit_transform(labels)
    return image_pixels, labels



def set_placeholders():
    X = tf.placeholder(tf.float32, [None, 500, 500,3])
    Y = tf.placeholder(tf.float32, [None, 5])

    return X, Y


def allocate_weights_and_biases(n_classes):

    weights = {
        'wc1': tf.get_variable('w0', shape=(3, 3, 3, 32), initializer=tf1.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf1.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf1.contrib.layers.xavier_initializer()),
        'wd1': tf.get_variable('W3', shape=(63 * 63 * 128, 300), initializer=tf1.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W6', shape=(300, n_classes), initializer=tf1.contrib.layers.xavier_initializer()),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf1.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf1.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf1.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B3', shape=(300), initializer=tf1.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B4', shape=(5), initializer=tf1.contrib.layers.xavier_initializer()),
    }

    return weights, biases


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


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

def loss_calculater_and_optimiser(learning_rate, batch_size, epochs, training_images, training_labels,
                                  test_images, test_labels, out, X, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Check whether the index of the maximum value of the predicted image is equal to the actual
    # labelled image and both will be a column vector.
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))

    # Calculate accuracy across all the given images and average them out.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        saver = tf.train.Saver()
        for i in range(epochs):
            for batch in range(len(training_images)//batch_size):
                batch_x = training_images[batch*batch_size:min((batch+1)*batch_size,len(training_images))]
                batch_y = training_labels[batch*batch_size:min((batch+1)*batch_size,len(training_labels))]
                # Run optimization op (backprop).
                # Calculate batch loss and accuracy
                opt = sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                print("Optimization Finished!")

                val_acc,valid_loss = sess.run([accuracy,cost], feed_dict={X: test_images, Y: test_labels})
                train_loss.append(loss)
                test_loss.append(valid_loss)
                train_accuracy.append(acc)
                test_accuracy.append(val_acc)
                print("Validation Accuracy:","{:.5f}".format(val_acc))
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver.save(sess, 'face-shape-classifier-model1')
    summary_writer.close()

    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()

    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


def restore_model(X_test, Y_test):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('face-shape-classifier-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    pred = graph.get_tensor_by_name("accuracy:0")
    x = graph.get_tensor_by_name("X:0")
    y_true = graph.get_tensor_by_name("Y:0")
    X_test = (X_test - np.min(X_test))/ (np.max(X_test) - np.min(X_test))
    feed_dict_testing = {x: X_test, y_true: Y_test}
    result=sess.run(pred, feed_dict=feed_dict_testing)
    print(result)



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


    args = vars(arg_parser.parse_args())
    image_directory = args['img_dir']
    labels_file = args['labels_file']
    landmarks_file = args['landmarks_file']
    face_shape_index = args['face_shape_index']
    preprocessed_data_file = args['preprocessed_data_file']
    print("Building face shape classification model from images in {},"
          " using labels file at {} and face landmarks file at {}. Index of face shape field in the CSV file is"
          " {}.".format(image_directory,labels_file, landmarks_file, face_shape_index))

    train_labels_file = 'face_shape_train.csv'
    X_train, Y_train = load_images(image_directory, face_shape_index, train_labels_file)

    val_labels_file = 'face_shape_val.csv'
    X_val, Y_val = load_images(image_directory, face_shape_index, val_labels_file)

    test_labels_file = 'face_shape_test.csv'
    X_test, Y_test = load_images(image_directory, face_shape_index, test_labels_file)

    X_train = (X_train - np.min(X_train))/ (np.max(X_train) - np.min(X_train))
    X_val = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val))
    X1, Y1 = set_placeholders()
    weights, biases= allocate_weights_and_biases(5)
    pred = conv_net(X1, weights, biases)
    loss_calculater_and_optimiser(0.0001, 90, 15, X_train, Y_train, X_val, Y_val, pred, X1, Y1)

    restore_model(X_test, Y_test)

