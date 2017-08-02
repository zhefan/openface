#!/usr/bin/env python2

import time
import math

start = time.time()

import argparse
import cv2
import os
import pickle
import sys

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

import rospy
import actionlib
from control_msgs.msg import PointHeadAction, PointHeadGoal
import roslib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    # print (reps)
    return (reps,bb)


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
        else:
                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    reps,bb = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print ("No Face detected")
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print (predictions)
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences, bb)


class PointHeadClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, max_v, duration=0.5):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        goal.max_velocity = max_v
        self.client.send_goal(goal)
        #self.client.wait_for_result()


def process_image(frame, args):
    confidenceList = []

    persons, confidences, bb = infer(frame, args)
    print ("P: " + str(persons) + " C: " + str(confidences))
    try:
        # append with two floating point precision
        confidenceList.append('%.2f' % confidences[0])
    except:
        # If there is no face detected, confidences matrix will be empty.
        # We can simply ignore it.
        pass

    counter = 0
    for i, c in enumerate(confidences):
        box = bb[counter]
        counter += 1

        cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), (255,0,0), 2 )
        
        x_offset = args.width/2.0 - box.center().x
        y_offset = args.height/2.0 - box.center().y
        print( 'height: '+str(args.height)+", y_center: "+str(box.center().y) )
        print( 'x_offset: '+str(x_offset)+", y_offset: "+str(y_offset) )

        x_comm = x_offset / 3000.0
        y_comm = y_offset / 2000.0

        if math.fabs(x_offset) < args.width/16:
            x_comm = 0.0

        if math.fabs(y_offset) < args.height/16:
            y_comm = 0.0

        print( 'x_comm: '+str(x_comm)+", y_comm: "+str(y_comm) )
        if args.test == 0:
            head_action.look_at(0.0, x_comm, y_comm, "head_tilt_link", 0.2)
        
        if c <= args.threshold:  # 0.5 is kept as threshold for known face.
            persons[i] = "_unknown"

    # Print the person name and conf value on the frame
    cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('', frame)
    cv2.waitKey(1)


'''Capture image from robot cam
'''
class image_converter:

    def __init__(self, input_args):
        #self.image_pub = rospy.Publisher("head_camera/rgb/image_raw",Image)
        self.args = input_args
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("head_camera/rgb/image_raw",Image,self.callback)


    def callback(self,data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame = cv2.resize(frame, (self.args.width, self.args.height))
            process_image(frame, self.args)
    
        except CvBridgeError as e:
            print(e)


'''Capture image from webcam
'''
def webcam(args):
    video_capture = cv2.VideoCapture(args.device)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    while True:
        ret, frame = video_capture.read()
        process_image(frame, args)
       # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()


   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='Capture device. 0 for webcam cam and 1 for robot cam')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    parser.add_argument('--test', type=int, default=0)

    
    args = parser.parse_args()
    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    rospy.init_node('head_chatter')
    if args.test == 0:
        head_action = PointHeadClient()
    # 
    if args.device == 0:
        webcam(args)
    else:
        ic = image_converter(args)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")


    cv2.destroyAllWindows()


