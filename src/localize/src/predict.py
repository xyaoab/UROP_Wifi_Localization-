#!/usr/bin/env python
import rospy
import os
import sys
from std_msgs.msg import String
from data import DataLoader
from geometry_msgs.msg import PoseStamped
import numpy as np 
from model import Model
from rvizMarker import RvizMarker

class listener:
    def __init__(self):
        self.user_sub = rospy.Subscriber("wifi/bssid", String, self.sub_callback)
        self.goal_pub = rospy.Publisher("move_base_simple/goal",PoseStamped,queue_size=1)
        directory = os.path.dirname(os.path.realpath(__file__))
        file_name = directory + '/training_wifi.csv'
        self.d = DataLoader(file_name)
        train_pos = self.d.get_train()[0]
        train_strength = self.d.get_train()[1]
        self.model_list = Model(train_pos,train_strength)
        self.model_list.train()
        self.model_list.grid(np.amin(train_pos,axis=0),np.amax(train_pos,axis=1))
        
        self.marker = RvizMarker('map', '/rviz/marker')
        self.marker.setDefaultMarkerParams()


    def sub_callback(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        #input_data = np.fromstring(data.data, dtype=str)
        input_data = np.array(data.data.split('\t'))
        input_bssid = input_data[0::2] 
        input_rssi = input_data[1::2].astype(np.float32)
        print(input_rssi)
        strength = self.d.get_user(input_bssid, input_rssi)
        print(strength)
        self.model_list.bayes_filter(self.model_list.belief, strength)


        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = self.model_list.predict_pos[0]
        target.pose.position.y = self.model_list.predict_pos[1]
        target.pose.position.z = 0
        target.pose.orientation.w = 1
        rospy.loginfo("target publish is %d, %d", target.pose.position.x, target.pose.position.y)
        self.goal_pub.publish(target)

        self.marker.publishMarker(target.pose)
    
def main(args):
    rospy.init_node('listener_android', anonymous=True)
    lis = listener()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)

