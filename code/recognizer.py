from vidutils import measure, compare
from tracking_hand import optical_flow_magnitude
from collections import defaultdict
import numpy as np
import cv2

# Gestures which have training data.
GESTURES = ('a', 'b', 'c')


class GestureRecognizer(object):
    """An abstract base recognizer class."""

    def train(self):
        raise NotImplementedError()

    def recognize_gesture(self, source):
        raise NotImplementedError()


class DifferenceRecognizer(GestureRecognizer):
    """A very simple attempt at gesture recognition. This class calculates
    pixel-by-pixel differences between energy images and selects the gesture
    with the smallest overall difference.
    """

    def __init__(self):
        # optical flow library of known gestures
        self.known_gestures = defaultdict(list)

    def train(self):
        """Reads in preprocessed information on gestures."""

        for gesture in GESTURES:
            # read data file in data/
            datafile = 'data/{0}.npz'.format(gesture)
            # load training data
            data = np.load(datafile)
            self.known_gestures[gesture] = [data[k] for k in data]

    # compare optical flows of unknown gestures with library
    def recognize_gesture(self, source):
        # Calculate the optical flow magnitute of unknown gesture
        source_flow = measure(source, f=optical_flow_magnitude())
        # resize to compare
        source_flow = cv2.resize(source_flow, (180, 320))

        avg_diffs = {}
        for gesture in GESTURES:
            # compare the unknown optical flow with data
            diffs = [simple_difference(source_flow, flow)
                     for flow in self.known_gestures[gesture]]
            avg_diffs[gesture] = np.mean(diffs)

        print avg_diffs
        # return the smallest difference -> the predicted gesture
        return min(GESTURES, key=lambda x: avg_diffs[x])


def simple_difference(a, b):
    ''' function used to compare two optical flows'''

    h, w = a.shape
    element_diff = np.subtract(a, cv2.resize(b, (w, h)))
    return np.sum(np.absolute(element_diff))


def match_score(a, b):
    ''' use matchTemplate to compare two optical flows '''
    h, w = a.shape
    b = cv2.resize(b, (w, h))
    return np.amin(cv2.matchTemplate(a, b, method=cv2.cv.CV_TM_SQDIFF_NORMED))


class SVMRecognizer(GestureRecognizer):
    ''' Use SVM for training and recogniting '''

    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.SVM()
        self.params = dict(kernel_type=cv2.SVM_RBF,
                           svm_type=cv2.SVM_C_SVC,
                           C=C,
                           gamma=gamma)

    def train(self):
        # list used to save names of gestures
        labels = []
        # data for a gesture (training)
        examples = []

        # loading data to training machine
        for i, gesture in enumerate(GESTURES):
            datafile = 'data/{0}.npz'.format(gesture)
            data = np.load(datafile)

            for k in data:
                labels.append(i)
                examples.append(data[k])

        self.model.train(
            np.array(examples), np.int32(labels), params=self.params)

    def recognize_gesture(self, source):
        # calculating the optical flow of unknown gesture
        # and reside to match with training data's
        source_flow = measure(source, f=optical_flow_magnitude())
        source_flow = cv2.resize(source_flow, (180, 320)).flatten()
        # getting the prediction after passing the unknown data
        # to SVM
        i = int(self.model.predict(source_flow))
        return GESTURES[i]


if __name__ == '__main__':
    # creating svm's object
    gr = SVMRecognizer()
    # training
    gr.train()

    # passing the unknown gestures to svm and return the results
    msg = 'Expected {0} got {1}'
    for gesture in GESTURES:
        source = 'videos/{0}0_blackbackground.mov'.format(gesture)
        print msg.format(gesture, gr.recognize_gesture(source))
