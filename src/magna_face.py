#!/usr/bin/env python2

import os
import argparse
import cv2
import rospy
import actionlib

import process_frame
import openface
import point_head
import face_recognition.msg


class HeadMover(object):
    """ head mover main class """
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
        """init face align lib and network"""
        if goal.comm:
            args.id = goal.id
            args.align_lib = openface.AlignDlib(args.dlibFacePredictor)
            args.net = openface.TorchNeuralNet(
                args.networkModel,
                imgDim=args.imgDim,
                cuda=args.cuda)
            args.head_action = point_head.PointHeadClient()

            # self._server.accept_new_goal()
            # if self._server.is_preempt_requested():
            #     rospy.loginfo('%s: Preempted' % self._action_name)
            #     self._as.set_preempted()
            #     self._rate.sleep()
            self._result = True
            if args.device == 0:
                process_frame.webcam(args)
            else:
                process_frame.image_converter(args)
                try:
                    pass
                except KeyboardInterrupt:
                    print("Shutting down")

            self._server.set_succeeded(True)

            if not args.noviz:
                cv2.destroyAllWindows()
        else:
            self._server.set_succeeded(False)
            self._result = False
            self._server.publish_feedback(self._result)


if __name__ == '__main__':
    """ main function """
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
    parser.add_argument('--noviz', action='store_true')
    parser.add_argument(
        '--classifierModel',
        type=str,
        help='The Python pickle representing the classifier. \
        This is NOT the Torch network model, which can be set with --networkModel.')
    parser.add_argument('--test', type=int, default=0)

    args = parser.parse_args()

    ######################################

    rospy.init_node('head_mover')
    server = HeadMover(rospy.get_name())
    rospy.spin()
