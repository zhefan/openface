import actionlib
import rospy
from face_recognition.msg import head_move

from control_msgs.msg import PointHeadAction, PointHeadGoal

client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)

def look_at(x, y, z, frame, max_v, duration=1.0):
    goal = PointHeadGoal()
    goal.target.header.stamp = rospy.Time.now()
    goal.target.header.frame_id = frame
    goal.target.point.x = x
    goal.target.point.y = y
    goal.target.point.z = z
    goal.min_duration = rospy.Duration(duration)
    goal.max_velocity = max_v
    client.send_goal(goal)
    #client.wait_for_result()
    client.stop_tracking_goal()

def callback(data):
    print data.x, data.y
    look_at(0.0, data.x, data.y, "head_tilt_link", 10.0)

if __name__ == "__main__":
    # Create a node
    rospy.init_node('head_listener', anonymous=True)
    
    rospy.loginfo("Waiting for head_controller...")
    client.wait_for_server()
    
    rospy.Subscriber("head_chatter", head_move, callback)
    
    
    rospy.spin()




