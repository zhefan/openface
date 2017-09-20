### Training
# align face
for N in {1..4}; do ./util/align-dlib.py data/lab/raw align outerEyesAndNose data/lab/aligned --size 96 & done

# generate feature
./batch-represent/main.lua -outDir /home/zhefanye/at_ws/src/face_recognition/data/lab/feat -data /home/zhefanye/at_ws/src/face_recognition/data/lab/aligned

# train the classifier
./demos/classifier.py train data/lab/feat

### Testing
# use flag --test to enable test mode (1 for testing without head action, 0 for deploying)
#python src/ros_face.py --classifierModel data/lab/feat/classifier.pkl --cuda --device 0 --id yzf --test 1 --verbose
roslaunch fetch_gazebo simulation.launch # connect to robot or run simulation
python src/magna_face.py --classifierModel data/lab/feat/classifier.pkl --cuda --device 1 --verbose

# starting another terminal
rostopic pub /head_mover/goal face_recognition/HeadMoverActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  id: 'yzf'
  comm: false" 


# optional single image testing
./demos/classifier.py infer data/lfw-subset/feat/classifier.pkl images/examples/{carell,adams,lennon}*
