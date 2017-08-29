# align face
for N in {1..4}; do ./util/align-dlib.py data/lab/raw align outerEyesAndNose data/lab/aligned --size 96 & done

# generate feature
./batch-represent/main.lua -outDir /home/zhefanye/at_ws/src/face_recognition/data/lab/feat -data /home/zhefanye/at_ws/src/face_recognition/data/lab/aligned

# train the classifier
./demos/classifier.py train data/lab/feat

# use flag --test to enable test mode (1 for testing without head action, 0 for deploying)
python src/ros_face.py --classifierModel data/lab/feat/classifier.pkl --cuda --verbose --test 1

# optional single image testing
./demos/classifier.py infer data/lfw-subset/feat/classifier.pkl images/examples/{carell,adams,lennon}*
