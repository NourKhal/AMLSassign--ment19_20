import argparse
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import A1.lab2_landmarks as l2


def transform_images_to_features_data(csv_file_path, gender_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, gender_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


def split_training_test_data(X, Y, testsize):
    return train_test_split(X, Y, test_size=testsize)


# apply grid search and cross validation to find the optimal hyperparameter combination of the training model
def img_SVM_GS_CV(training_images, training_labels, test_images, test_labels):
    parameter_candidates = [
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['poly']},
    ]
    scores = ['precision', 'recall', 'f1-score', 'support']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(estimator=SVC(), cv=3, param_grid=parameter_candidates)
        clf.fit(training_images, training_labels)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_labels, clf.predict(test_images)
    print(classification_report(y_true, y_pred))
    print()

# sklearn functions implementation
# SVM model with the optimal hyperparameters combination found by gridsearch and cross validation
def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = SVC(kernel='rbf', gamma=0.0001, C=100)
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
    arg_parser.add_argument('-pd', '--preprocessed-data-file',
                            help='the path to the preprocessed image data file',
                            required=True)

    args = vars(arg_parser.parse_args())
    image_directory = args['img_dir']
    labels_file = args['labels_file']
    landmarks_file = args['landmarks_file']
    gender_index = args['gender_index']
    preprocessed_data_file = args['preprocessed_data_file']
    print("Building gender classification model from images in {}, using extracted image features at {},"
          " of labels file at {} and face landmarks file at {}. Index of smiling field in the CSV file is"
          " {}".format(image_directory, preprocessed_data_file, labels_file, landmarks_file, gender_index))

    filename = preprocessed_data_file

    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)

    # split the data into 70% taiining, 15% validation and 15% test sets

    X_train, X_test, Y_train, Y_test = split_training_test_data(X, Y, 0.3)
    X_test, X_val, Y_test, Y_val = split_training_test_data(X_test, Y_test, 0.5)

    # img_SVM_GS_CV is used to train the model
    pred = img_SVM_GS_CV(X_train.reshape((3360, 68*2)), list(zip(*Y_train))[0], X_val.reshape((720, 68*2)), list(zip(*Y_val))[0])

    # img_SVM is used to test the model
    pred_test = img_SVM(X.reshape((4800, 68*2)), list(zip(*Y))[0], X_test.reshape((720, 68*2)), list(zip(*Y_test))[0])



