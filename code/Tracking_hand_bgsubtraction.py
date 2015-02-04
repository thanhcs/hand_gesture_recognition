import cv2
import numpy
from vidutils import _vid_capture

PRIMARY_CAMERA = 0


def processing(source=None):
    # use the webcam if no source input.
    if source is None:
        source = PRIMARY_CAMERA

    # background subtractor
    fgbg = cv2.BackgroundSubtractorMOG2()
    # pos_x = 0
    # pos_y = 0
    # fixed_position = false

    with _vid_capture(source) as cap:

        # getting the background in 5 frames
        for i in xrange(5):
            valid, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgbg.apply(gray)

        print "I'm ready."

        while (valid):
            # read...
            valid, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # loading to background subtractor object
            fgmask = fgbg.apply(frame)
            # display
            cv2.imshow('video', frame)

            # Should I use a box that put the hand in there?
            # cv2.rectangle(frame,(150,150),(350,350),(255,0,255),2)

            # Reduce noise
            ret, thres = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)
            # blur
            blur = cv2.medianBlur(thres, 5)
            # find contours
            contours, hierarchy = cv2.findContours(
                blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # get the largest contour
                cnt = max(contours, key=cv2.contourArea)
                # draw contours
                cv2.drawContours(blur, cnt, -1, (255, 0, 255), 3)

                # get the max values of the box around the contours
                # x,y,w,h = cv2.boundingRect(cnt)
                # cv2.rectangle(blur,(x,y),(x+w,y+w+100),(255,0,255),2)

                # choose the position that's comfortable with the hand
                # if cv2.waitKey(1) & 0xFF == ord('s'):
                #     pos_x = x
                #     pos_x = y
                #     fixed_position = True

                cv2.imshow('subtracted', blur)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # user interaction
    raw_input("Press enter when you're ready. \
        \nThere shouldn't be anything moving around at this time")
    print "Processing..."

    processing()
