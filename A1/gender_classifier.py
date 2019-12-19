import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import A1.lab2_landmarks as l2


def transform_images_to_features_data(csv_file_path, gender_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, gender_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, train_size=0.8)
    return tr_X, tr_Y, te_X, te_Y

# sklearn functions implementation


def img_SVM(training_images, training_labels, test_images, test_labels):

    classifier = SVC(kernel='poly')
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

    # print(pred)
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

    tr_X, tr_Y, te_X, te_Y = transform_images_to_features_data(labels_file, gender_index, image_directory, landmarks_file)

    pred = img_SVM(tr_X.reshape((3840, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((960, 68*2)), list(zip(*te_Y))[0])
