import argparse
import pickle

import numpy as np

import A1.lab2_landmarks as l2


def transform_images_to_features_data(csv_file_path, smiling_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, smiling_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


def split_train_test_data(x, y, test_size):
    return split_train_test_data(x, y, test_size)



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
    emotion_index = args['emotion_index']
    preprocessed_data_file = args['preprocessed_data_file']
    print("Building gender classification model from images in {},"
          " using labels file at {} and face landmarks file at {}. Index of smiling field in the CSV file is"
          " {}. The image features are written to a pickle file {}".format(image_directory,labels_file, landmarks_file,
                                                                           emotion_index,
                                                                           preprocessed_data_file))

    X, Y = transform_images_to_features_data(labels_file, emotion_index, image_directory, landmarks_file)
    x_y = list(zip(X, Y))

    pickled = (X, Y)
    filename = preprocessed_data_file

    with open(filename, 'wb') as f:
        pickle.dump(pickled, f)
