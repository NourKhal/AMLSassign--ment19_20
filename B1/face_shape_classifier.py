import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import A1.lab2_landmarks as l2
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def transform_images_to_features_data(csv_file_path, smiling_column_index, image_dir, face_landmarks_path):

    X, y = l2.extract_features_labels(csv_file_path, smiling_column_index, image_dir, face_landmarks_path)
    Y = np.array([y, -(y - 1)]).T
    return X, Y


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

    X, Y = transform_images_to_features_data(labels_file, face_shape_index, image_directory, landmarks_file)
    x_y = list(zip(X, Y))

    pickled = (X, Y)
    filename = preprocessed_data_file

    with open(filename, 'wb') as f:
        pickle.dump(pickled, f)

    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)

    