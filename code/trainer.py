"""Project 4: Hand gesture recognition
"""

import cv2
import numpy as np
import sys
from collections import deque
import datetime
import os

CAP_WIN_WIDTH = 640
CAP_WIN_HEIGHT = 480
FLOW_WIN_WIDTH = 320
FLOW_WIN_HEIGHT = 240
FRAME_INSET_X = 80
FRAME_INSET_Y = 20

FLOW_THRESHOLD = 4.0
NUM_ENERGY_IMAGE_FRAMES = 20
ENERGY_IMAGE_SCALE = 0.03
NUM_FRAMES_DELAY_TO_CAPTURE = 20

BGR_GREEN = (0, 255, 0)
BGR_RED = (0, 0, 255)
BGR_YELLOW = (0, 255, 255)

GESTURE_NAMES = ['NONE', 'WAVE', 'AIRQUOTES', 'PUNCH', 'POINT']
GESTURE_LABELS = [1, 2, 3, 4, 5]


def _draw_flow(img, flow, step=8):
    """Draw flow vectors over input image.  Copied from opt_flow.py
    example in opencv library.

    Arguments:
      arg1: arg1 description
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def _create_energy_images(energy_path):
    """Create the energy images on user demand.  User presses a key from
    1 to 5 and an energy image is datestamped and saved.

    Arguments:
      energy path: path to output dir for energy images
    """
    assert energy_path.endswith('/')

    # Init the video capture
    cap = cv2.VideoCapture(0)

    # At the moment we only support 640x480 cameras
    assert int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)) == CAP_WIN_WIDTH
    assert int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) == CAP_WIN_HEIGHT

    # Create energy image and history list
    energy_history = deque()

    # Do training loop
    ret, frame = cap.read()
    assert ret
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_flow = cv2.resize(gray, (FLOW_WIN_WIDTH, FLOW_WIN_HEIGHT))

    capture_delay = 0
    capture_key = 0

    while(True):
        # Calculate the optical flow
        ret, frame = cap.read()
        assert ret
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_flow = cv2.resize(gray, (FLOW_WIN_WIDTH, FLOW_WIN_HEIGHT))
        # Params: prev, next, pyr_scale, levels, winsize, iterations,
        # poly_n, poly_sigma, flags
        flow = cv2.calcOpticalFlowFarneback(
            prev_flow, next_flow, 0.5, 1, 10, 1, 5, 1.2, 0)
        flow[flow < FLOW_THRESHOLD] = 0.0
        prev_flow = next_flow

        # Calculate magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Update energy image
        energy_history.append(mag)
        if len(energy_history) > NUM_ENERGY_IMAGE_FRAMES:
            energy_history.popleft()
        energy_image = np.zeros((FLOW_WIN_HEIGHT, FLOW_WIN_WIDTH), np.float32)
        delta_scale = 1.0 / len(energy_history)
        scale = delta_scale
        for i in energy_history:
            energy_image += i * scale
            scale += delta_scale

        # Capture energy image
        if capture_delay > 0:
            capture_delay -= 1
            if capture_delay == 0:
                # Store energy image to disk
                out_file = (energy_path + GESTURE_NAMES[int(capture_key) - 1]
                            + '_' +
                            datetime.datetime.now().strftime("D%Y%m%dT%H%M%S"))
                np.save(out_file + '.npy', energy_image)
                cv2.imwrite(out_file + '.png', energy_image * 20.0)

        # Display the input frame with valid input area
        color = BGR_RED if capture_delay > 0 else BGR_GREEN
        cv2.rectangle(frame,
                      (FRAME_INSET_X, FRAME_INSET_Y),
                      (CAP_WIN_WIDTH - FRAME_INSET_X,
                       CAP_WIN_HEIGHT - FRAME_INSET_Y),
                      color, 3)
        cv2.imshow('input', frame)

        # Display the flow frame
        cv2.imshow('flow', _draw_flow(next_flow, flow))

        # Display the energy image
        cv2.imshow('energy', energy_image * ENERGY_IMAGE_SCALE)

        # Handle input
        wait_key = cv2.waitKey(1) & 0xFF
        if wait_key == 27:
            # ESC to exit program
            break
        elif wait_key >= ord('1') and wait_key <= ord('5'):
            capture_delay = NUM_FRAMES_DELAY_TO_CAPTURE
            capture_key = chr(wait_key)

    # Cleanup video capture
    cap.release()
    cv2.destroyAllWindows()


def _svm_train(svm_output_fn, energy_path):
    """Train the SVM classifier.

    Arguments:
      svm_output_fn: name of SVM classifier output file
    """
    assert energy_path.endswith('/')

    files = os.listdir(energy_path)

    # Create SVM
    svm = cv2.SVM()

    # Load energy files for each gesture
    labels = []
    examples = []
    test_labels = []
    test_examples = []

    for gesture_name, gesture_label in zip(GESTURE_NAMES, GESTURE_LABELS):
        all_files = [f for f in files if f.startswith(gesture_name) and
                     f.endswith('.npy')]

        gesture_files = all_files[:len(all_files)/2]
        for gesture_file in gesture_files:
            labels.append(gesture_label)
            energy_image = np.load(energy_path + gesture_file)
            examples.append(energy_image.flatten())

        test_files = all_files[len(all_files)/2:]
        for test_file in test_files:
            test_labels.append(gesture_label)
            energy_image = np.load(energy_path + test_file)
            test_examples.append(energy_image.flatten())

    # Train SVM
    svm_params = dict(kernel_type=cv2.SVM_POLY,  # cv2.SVM_RBF
                      svm_type=cv2.SVM_C_SVC,
                      degree=5,
                      C=1.0,
                      gamma=3.0)     # Default = 0.5
    svm.train(np.array(examples), np.array(labels), params=svm_params)

    # Test classifier and print stats
    correct = 0.0
    for example, label in zip(test_examples, test_labels):
        predicted_gesture_name = GESTURE_NAMES[int(svm.predict(example))-1]
        truth_gesture_name = GESTURE_NAMES[label-1]
        print ('predicted: ' + predicted_gesture_name +
               '  truth: ' + truth_gesture_name)
        if predicted_gesture_name == truth_gesture_name:
            correct += 1
    print 'Prediction accuracy: '+str((correct/len(test_examples))*100.0)+'%'

    # Save SVM trained data
    svm.save(svm_output_fn)


def _recognize(svm_input_fn):

    """Do gesture recognition using the SVM file created during training.

    Arguments:
      svm_input_fn: filename of SVM input file
    """
    # Init the video capture
    cap = cv2.VideoCapture(0)

    # At the moment we only support 640x480 cameras
    assert int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)) == CAP_WIN_WIDTH
    assert int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) == CAP_WIN_HEIGHT

    # Create SVM
    svm = cv2.SVM()
    svm.load(svm_input_fn)

    # Create energy history deque
    energy_history = deque()

    # Do recognition loop
    ret, frame = cap.read()
    assert ret
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_flow = cv2.resize(gray, (FLOW_WIN_WIDTH, FLOW_WIN_HEIGHT))

    while(True):
        # Calculate the optical flow
        ret, frame = cap.read()
        assert ret
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_flow = cv2.resize(gray, (FLOW_WIN_WIDTH, FLOW_WIN_HEIGHT))
        # Params: prev, next, pyr_scale, levels,
        # winsize, iterations, poly_n, poly_sigma, flags
        flow = cv2.calcOpticalFlowFarneback(
            prev_flow, next_flow, 0.5, 1, 10, 1, 5, 1.2, 0)
        flow[flow < FLOW_THRESHOLD] = 0.0
        prev_flow = next_flow

        # Calculate magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Update energy image
        energy_history.append(mag)
        if len(energy_history) > NUM_ENERGY_IMAGE_FRAMES:
            energy_history.popleft()
        energy_image = np.zeros((FLOW_WIN_HEIGHT, FLOW_WIN_WIDTH), np.float32)
        delta_scale = 1.0 / len(energy_history)
        scale = delta_scale
        for i in energy_history:
            energy_image += i * scale
            scale += delta_scale

        # Attempt to recognize
        label = int(svm.predict(energy_image.flatten()))

        # Display the input frame with valid input area and label
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, GESTURE_NAMES[
            int(label) - 1], (10, 100), font, 3, BGR_GREEN, 2)

        cv2.rectangle(frame, (FRAME_INSET_X, FRAME_INSET_Y),
                      (CAP_WIN_WIDTH - FRAME_INSET_X,
                       CAP_WIN_HEIGHT - FRAME_INSET_Y),
                      BGR_YELLOW, 3)
        cv2.imshow('input', frame)

        # Display the flow frame
        cv2.imshow('flow', _draw_flow(next_flow, flow))

        # Display the energy image
        cv2.imshow('energy', energy_image * ENERGY_IMAGE_SCALE)

        # Handle input
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    # Cleanup video capture
    cap.release()
    cv2.destroyAllWindows()


def _main():
    """Main function for training SVM classifier to recognize hand
    motion energy images.

    usage: python trainer.py svm_output energy_path
            (or)
           python trainer.py svm_input

    where:
      svm_output = filename of SVM classifier output file
      energy_path = path to energy images
        (or)
      svm_input = filename of SVM classifier input file
    """
    # Get command line arguments
    args = sys.argv
    args.pop(0)
    if len(args) < 1:
        print "\nERROR:\tIncorrect usage!\n"
        print "usage:\tpython trainer.py svm_output energy_path"
        print "\t  (or)"
        print "\tpython trainer.py svm_input\n"
        print "where:\tsvm_output = filename of SVM classifier output file"
        print "\tenergy_path = path to energy images"
        print "\t  (or)"
        print "\tsvm_input = filename of SVM input file\n"
        exit(0)

    if len(args) == 1:
        svm_input_fn = args.pop(0)
        # Do gesture recognition
        _recognize(svm_input_fn)
    else:
        svm_output_fn = args.pop(0)
        energy_path = args.pop(0)

        # Create the energy images
        _create_energy_images(energy_path)

        # Do the SVM training
        _svm_train(svm_output_fn, energy_path)

if __name__ == '__main__':
    _main()
