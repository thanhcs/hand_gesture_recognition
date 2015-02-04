import cv2
from contextlib import contextmanager
import numpy as np

PRIMARY_CAMERA = 0


@contextmanager
def _vid_capture(source=0):
    """Wrapper to manage access to a video feed or file.

    Arguments:
      source (optional): specifies either the name of a video file or the
        index of a camera to read from (default).
    """

    cap = cv2.VideoCapture(source)
    yield cap

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


def _vid_props(video):
    """Retrieves various video properties."""

    return {'fourcc': int(video.get(cv2.cv.CV_CAP_PROP_FOURCC)),
            'fps':    int(video.get(cv2.cv.CV_CAP_PROP_FPS)),
            'width':  int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            'height': int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))}


def measure(source, f):
    with _vid_capture(source) as cap:
        frames = iter(lambda: cap.read()[1], None)
        return reduce(lambda x, y: np.add(x, y), map(f, frames))


def compare(source=None, f=None, pip=True):
    """Displays a picture-in-picture video using the specified source or the
    computer's primary camera (default). The sub-picture optionally displays a
    version of the video processed using f.

    Arguments:
      source (optional): specifies the name of a video file to read.
      If omitted, defaults to primary camera.

      f (optional): a function which accepts a 3-channel frame from the video
      and returns a processed 3-channel copy of the frame.

      pip (optional): True to show picture-in-picture;
                    False to show side-by-side
    """

    if source is None:
        source = PRIMARY_CAMERA

    with _vid_capture(source) as cap:
        props = _vid_props(cap)
        w, h = props['width'], props['height']

        while(True):
            valid, frame = cap.read()

            if not valid:
                print "End of video."
                break

            # Get processed subframe and insert into upper left corner
            sub_frame = f(frame) if f is not None else np.copy(frame)
            if pip:
                sub_w, sub_h = w / 3, h / 3
                sub_frame = cv2.resize(sub_frame, (sub_w, sub_h))
                frame[0:sub_h, 0:sub_w] = sub_frame
            else:
                sub_frame = cv2.resize(sub_frame, (w, h))
                frame = np.hstack([frame, sub_frame])

            # Display the resulting frame
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print "Manual exit."
                break


if __name__ == '__main__':
    def to_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    compare(source='face.mov', f=to_gray)
