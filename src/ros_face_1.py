#!/usr/bin/env python2

import argparse
import cv2
import os
import rospy
import actionlib

import process_frame
import openface
import face_recognition.msg


class HeadMover(object):
    # create messages that are used to publish feedback/result
    _feedback = face_recognition.msg.HeadMoverFeedback()
    _result = face_recognition.msg.HeadMoverResult()

    def __init__(self, name):
        self._action_name = name
        self._rate = rospy.Rate(10)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            face_recognition.msg.HeadMoverAction,
            execute_cb=self.execute_cb,
            auto_start=False)
        self._server.start()

    def execute_cb(self, goal):
        # init face align lib and network
        align_lib = openface.AlignDlib(args.dlibFacePredictor)
        net = openface.TorchNeuralNet(
            args.networkModel,
            imgDim=args.imgDim,
            cuda=args.cuda)

        self._server.set_succeeded(True)

        # if self._server.is_preempt_requested():
        #     rospy.loginfo('%s: Preempted' % self._action_name)
        #     self._as.set_preempted()
        #     self._rate.sleep()

        if args.device == 0:
            process_frame.webcam(align_lib, net, args)
        else:
            ic = process_frame.image_converter(align_lib, net, args)
            try:
                pass
            except KeyboardInterrupt:
                print("Shutting down")

        cv2.destroyAllWindows()


''' main function
'''
if __name__ == '__main__':

    fileDir = os.path.dirname(os.path.realpath(__file__))
    modelDir = os.path.join(fileDir, '..', 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--id', type=str, help="id to track", default="zz")
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

    ######################################

    rospy.init_node('head_mover')
    server = HeadMover(rospy.get_name())
    rospy.spin()
