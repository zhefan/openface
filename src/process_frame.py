import cv2
import rospy
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import process_face
import point_head


'''Capture image from robot cam
'''
class image_converter:

    def __init__(self, align_lib, net, input_args):
        #self.image_pub = rospy.Publisher("head_camera/rgb/image_raw",Image)
        self.args = input_args
        self.align_lib = align_lib
        self.net = net
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("head_camera/rgb/image_raw",Image,self.callback)


    def callback(self,data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame = cv2.resize(frame, (self.args.width, self.args.height))
            process_image(frame, self.align_lib, self.net, self.args)
    
        except CvBridgeError as e:
            print(e)


'''Capture image from webcam
'''
def webcam(align_lib, net, args):
    video_capture = cv2.VideoCapture(args.device)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    while True:
        ret, frame = video_capture.read()
        process_image(frame, align_lib, net, args)
       # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()


'''Detect and recognize faces
'''
def process_image(frame, align_lib, net, args):
    x_flg = False
    y_flg = False

    confidenceList = []
    persons, confidences, bb = process_face.detect(frame, align_lib, net, args)
    if args.verbose:
        print ("P: " + str(persons) + " C: " + str(confidences))
    try:
        # append with two floating point precision
        confidenceList.append('%.2f' % confidences[0])
    except:
        # If there is no face detected, confidences matrix will be empty.
        # We can simply ignore it.
        pass

    for i, c in enumerate(confidences):
        box = bb[i]
        if persons[i] != args.id:
            cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), (255,0,0), 2 )

        elif persons[i] == args.id and c > args.threshold:
            cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), (0,0,255), 2 )

            x_offset = args.width/2.0 - box.center().x
            y_offset = args.height/2.0 - box.center().y
            if args.verbose:
                print( 'height: '+str(args.height)+", y_center: "+str(box.center().y) )
                print( 'x_offset: '+str(x_offset)+", y_offset: "+str(y_offset) )

            x_comm = x_offset / 3000.0
            y_comm = y_offset / 2000.0

            if math.fabs(x_offset) < args.width/16:
                x_comm = 0.0
                x_flg = True

            if math.fabs(y_offset) < args.height/16:
                y_comm = 0.0
                y_flg = True

            if args.verbose:
                print( 'x_comm: '+str(x_comm)+", y_comm: "+str(y_comm) )
            if args.test == 0:
                head_action = point_head.PointHeadClient()
                head_action.look_at(0.0, x_comm, y_comm, "head_tilt_link", 0.2)
                

    # Print the person name and conf value on the frame
    cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('', frame)
    cv2.waitKey(1)
    
    if x_flg and y_flg:
        return True
    else:
        return False    

