import numpy as np
from tracking_hand import *
import cv2
from vidutils import measure


def _process(gesture, idx, f):
    source = 'videos/{0}{1}_blackbackground.mov'.format(gesture, idx)
    print '\t', source
    data = measure(source, f)
    data = cv2.resize(data, (180, 320))
    return data.flatten()


if __name__ == '__main__':
    for gesture in ['a', 'b', 'c']:
        print 'Processing gesture:', gesture

        data = [_process(
            gesture, i, f=optical_flow_magnitude()) for i in xrange(10)]
        dest = 'data/{0}.npz'.format(gesture)

        print 'Writing to file:', dest
        np.savez(dest, *data)
