import argparse
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import A1.lab2_landmarks as l2


def transform_images_to_features_data(csv_file_path, gender_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, gender_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


def split_training_test_data(X, Y, testsize):
    return train_test_split(X, Y, test_size=testsize)

# sklearn functions implementation


def img_SVM(training_images, training_labels, test_images, test_labels):

    classifier = SVC(kernel='poly', gamma='scale')
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))
    print(classification_report(test_labels, pred))

    return pred


if __name__ == '__main__': ## command J

    arg_parser = argparse.ArgumentParser(description='ML model trained on images to predict gender from input images.')
    arg_parser.add_argument('-i', '--img-dir',
                            help='the path to the directory where the images are',
                            required=True)
    arg_parser.add_argument('-l', '--labels-file',
                            help='the path to the csv labels file',
                            required=True)
    arg_parser.add_argument('-s', '--landmarks-file',
                            help='the path to the face landmarks file',
                            required=True)
    arg_parser.add_argument('-gi', '--gender-index',
                            help='the index of the gender column in the labels.csv file',
                            required=True,
                            type=int)

    args = vars(arg_parser.parse_args())
    image_directory = args['img_dir']
    labels_file = args['labels_file']
    landmarks_file = args['landmarks_file']
    gender_index = args['gender_index']
    print("Building gender classification model from images in {}, using labels file at {} and face landmarks file at "
          "{}. Index of gender field in the CSV file is {}".format(image_directory,
                                                                   labels_file,
                                                                   landmarks_file,
                                                                   gender_index))
    # X, Y = transform_images_to_features_data(labels_file, gender_index, image_directory, landmarks_file)
    # x_y = list(zip(X, Y))
    #
    # pickled = (X, Y)
    filename = 'preprocessed_data1.pickle'
    #
    # with open(filename, 'wb') as f:
    #     pickle.dump(pickled, f)

    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5)
    pred = img_SVM(X_train.reshape((3360, 68*2)), list(zip(*Y_train))[0], X_test.reshape((720, 68*2)), list(zip(*Y_test))[0])
