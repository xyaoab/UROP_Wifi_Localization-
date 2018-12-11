#!/usr/bin/env python
import roslib
import rospy
import pywifi
import csv
import datetime
from nav_msgs.msg import Odometry

def odometryCb(odom_data):
    row = []
    curr_time = odom_data.header.stamp
    pose = odom_data.pose.pose #  the x,y,z pose and quaternion orientation
    print  curr_time
    print
    print pose
    row.append(curr_time)
    row.append(float(pose.position.x))
    row.append(float(pose.position.y))
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[1]
    iface.scan()
    result=iface.scan_results()
    #print('results after scanning')
    for i in range(len(result)):
        if(result[i].ssid == "eduroam"):
            #print(result[i].bssid, " ",result[i].signal)
            row.append(result[i].bssid)
            row.append(result[i].signal)
    with open("trainng_wifi.csv",'ab') as file:
        wtr = csv.writer(file)
        wtr.writerow(row)
    rospy.sleep(3)

if __name__ == "__main__":
    rospy.init_node('oodometry', anonymous=True) #make node 
    rospy.Subscriber('odom',Odometry,odometryCb,queue_size=1)
    rospy.spin()
