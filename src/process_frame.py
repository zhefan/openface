""" Magna demo core
"""
import math
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter(object):
    """ Capture image from robot cam """

    def __init__(self, input_args):
        self.args = input_args
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "head_camera/rgb/image_raw", Image, self.callback)
        self.frame = None

    def callback(self, data):
        """ RGB image call back """
        try:
            self.frame = cv2.resize(self.bridge.imgmsg_to_cv2(data, "bgr8"),
                                    (self.args.width, self.args.height))
        except CvBridgeError as e:
            print(e)

    def robot_process_img(self):
        """ dummy method """
        if self.frame is not None:  # sanity check
            return process_image(self.frame, self.args)
        else:
            return False


def webcam(args):
    """ Capture image from webcam """
    video_capture = cv2.VideoCapture(args.device)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    while True:
        _, frame = video_capture.read()
        process_image(frame, args)
       # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()


def process_image(frame, args):
    """ Detect and recognize faces """
    x_flg = False
    y_flg = False

    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get the largest face bounding box
    box = args.align_lib.getLargestFaceBoundingBox(rgbImg)  # Bounding box
    if box is not None:  # if face found
        x_offset = args.width / 2.0 - box.center().x
        y_offset = args.height / 2.0 - box.center().y
        if args.verbose:
            print('height: ' + str(args.height) +
                  ", y_center: " + str(box.center().y))
            print('x_offset: ' + str(x_offset) +
                  ", y_offset: " + str(y_offset))

        x_comm = x_offset / 3000.0
        y_comm = y_offset / 2000.0

        if math.fabs(x_offset) < args.width / 16:
            x_comm = 0.0
            x_flg = True

        if math.fabs(y_offset) < args.height / 16:
            y_comm = 0.0
            y_flg = True

        if args.verbose:
            print('x_comm: ' + str(x_comm) + ", y_comm: " + str(y_comm))

        # send command to robot
        args.head_action.look_at(
            0.0, x_comm, y_comm, "head_tilt_link", 0.2)

        # display red bbox if face id found
        if not args.noviz:
            cv2.rectangle(frame, (box.left(), box.top()),
                          (box.right(), box.bottom()), (0, 0, 255), 2)

    else:
        pass

    if not args.noviz:
        cv2.imshow('', frame)
        cv2.waitKey(1)

    if x_flg and y_flg:
        return True
    else:
        return False
