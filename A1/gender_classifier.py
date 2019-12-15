import os
import A1.lab2_landmarks as l2
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def transform_data():

    X, y = l2.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X, tr_Y, te_X, te_Y = train_test_split(X, Y, train_size=0.8)

    return tr_X, tr_Y, te_X, te_Y

# sklearn functions implementation


def img_SVM(training_images, training_labels, test_images, test_labels):

    classifier = SVC(kernel='linear')
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

    # print(pred)
    return pred


tr_X, tr_Y, te_X, te_Y = transform_data()

pred=img_SVM(tr_X.reshape((100, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((35, 68*2)), list(zip(*te_Y))[0])
