"""This file is for couting problem (step 2).

Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
"""
import numpy as np   # We recommend to use numpy arrays
from sklearn.base import BaseEstimator
import skimage.filters as filters


class model_count(BaseEstimator):
    """Main class for Couting problem."""

    def __init__(self, clf_model):
        """Init method.

        This constructor is supposed to initialize data members.
        It takes as input the binary classification model of patches.
        (You can use it for fit and predict method)
        """
        self.clf_model = clf_model  # Trained model from patch classification

    def fit(self, X, y):
        """Fit method.

        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (750, 750, 3) then 1687500 features.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        """
        self.num_train_samples = X.shape[0]
        X = X.reshape((self.num_train_samples, 750, 750, 3))

        self.is_trained = True

    def predict(self, X):
        """Predict method.

        This function should provide predictions of labels on (test) data.
        The function predict the number of parasites in an image.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (750, 750, 3) then 1687500 features.
        It is a simple sliding widow of the current classification model.
        To aggregate information from window,
        we use non maximum suppression method.
        """
        step = 15  # step sliding window
        size = 40
        self.num_train_samples = X.shape[0]
        X = X.reshape((self.num_train_samples, 750, 750, 3))

        pred = []

        for img in X:  # Loop to predict the number of parasites in each image
            p = self.predict_proba(img, step, size)
            boxes = self._get_boxes(img, p, step, size, threshold=0.5)
            found = self.non_maximum_suppression(boxes, overlapThresh=0.3)
            pred.append(len(found))

        return pred

    def predict_proba(self, img, step, size):
        """Predict method.

        The function predict the probability of being positive for all patches
        extrated from one image with fixed step.
        Args:
            img: Image (750, 750, 3).
            step: step for sliding window
            size: size of the patch
        """
        height, width, channels = img.shape

        probs = np.zeros((int(img.shape[0]*1.0/step),
                          int(img.shape[1]*1.0/step)))
        patches = []

        y = 0
        while y+(size) < height:
                    x = 0
                    predictions = []
                    while (x + size < width):
                        left = x
                        right = x+(size)
                        top = y
                        bottom = y+(size)
                        patches.append(img[top:bottom, left:right, :])
                        x += step
                    y += step

        p = np.array(patches)
        predictions = self.clf_model.predict(p.reshape((p.shape[0], 40*40*3)))
        i = 0
        y = 0
        while y+(size) < height:
                    x = 0
                    while (x+(size) < width):
                        left = x
                        right = x+(size)
                        top = y
                        bottom = y+(size)
                        probs[int(y/step), int(x/step)] = predictions[i]
                        i += 1
                        x += step
                    y += step

        return probs

    def _get_boxes(self, img, probs, step, size, threshold=0.5):
        probs = filters.gaussian(probs, 1)

        height, width, channels = img.shape
        boxes = []

        i = 0
        y = 0
        while y+(size) < height:
                    x = 0
                    while (x+(size) < width):
                        left = int(x)
                        right = int(x+(size))
                        top = int(y)
                        bottom = int(y+(size))
                        if probs[int(y/step), int(x/step)] > threshold:
                            boxes.append([left, top, right, bottom,
                                          probs[int(y/step), int(x/step)]])
                        i += 1
                        x += step
                    y += step

        if len(boxes) == 0:
            return np.array([])

        boxes = np.vstack(boxes)
        return boxes

    def non_maximum_suppression(self, boxes, overlapThresh):
        """Malisiewicz et al.

        Python port by Adrian Rosebrock
        """
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                             np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")
