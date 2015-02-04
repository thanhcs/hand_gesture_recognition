from vidutils import compare, measure
import cv2
import numpy as np
from collections import deque
import sys


def skin_mask():
    ''' trying to use skin recognition to detect the hand '''

    def process(frame, windowSize=5):

        lower = np.array([0, 30, 40], dtype="uint8")
        upper = np.array([60, 255, 255], dtype="uint8")

        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        skinMask = cv2.inRange(converted, lower, upper)
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        return skin  # cv2.cvtColor(skin, cv2.COLOR_GRAY2BGR)

    return process


def background_subtractor():
    ''' trying to use background subtractor '''

    sub = cv2.BackgroundSubtractorMOG()

    def process(frame):
        result = sub.apply(frame, learningRate=-1)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return process


def optical_flow():
    flow = optical_flow_magnitude()

    def process(frame):
        mag = flow(frame)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

    return process


def optical_flow_magnitude():
    ''' calculate the magnitude of an optical flow '''
    d = {'prev': None}

    def process(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.pyrDown(frame)

        if d['prev'] is None:
            d['prev'] = frame
        # Computes a dense optical flow
        # using the Gunnar Farneback’s algorithm
        flow = cv2.calcOpticalFlowFarneback(d['prev'],
                                            frame,
                                            pyr_scale=0.5,
                                            levels=1,
                                            winsize=10,
                                            iterations=1,
                                            poly_n=5,
                                            poly_sigma=1.1,
                                            flags=0)
        # Calculates the magnitude and angle of 2D vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # threshold it
        _, mag = cv2.threshold(mag, 3, 255, cv2.THRESH_TOZERO)
        # store the current frame
        d['prev'] = frame

        return mag

    return process


def convex_hull():

    window = deque()

    def process(frame, windowSize=10):
        # get gray scale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # blur it
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold to get the clear black and white frame
        ret, thres = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

        window.append(thres)
        if len(window) > windowSize:
            window.popleft()

        avg_thres = sum(window) / len(window)

        contours, hierarchy = cv2.findContours(
            avg_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # get the largest contour
            cnt = max(contours, key=cv2.contourArea)

            # find the convex hull of the hand
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            box = np.int32(cv2.cv.BoxPoints(rect))

            # draw a box around the hand
            drawing = np.zeros(frame.shape, np.uint8)
            cv2.drawContours(drawing, [cnt], 0, (0, 0, 255), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [box], 0, (255, 0, 0), 2)

            return drawing
        else:
            return frame

    return process

if __name__ == '__main__':
    i = int(sys.argv[1])
    if i == 0:
        # use convex hull method if input parameter is 1
        func = convex_hull()
    else:
        # use skin recognition otherwise
        func = skin_mask()

    for x in ['a', 'b', 'c', 'l']:
        compare(source='videos/{0}.mov'.format(x), f=func, pip=False)
