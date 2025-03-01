#!/usr/bin/env python3
from drone_pubsub import *

def main():
    print("Drone2 Node")
    rospy.init_node('drone2_node')
    drone2 = PubSub("current2_x", "current2_y", "current2_z", "roll2", "pitch2",
                    "yaw2", "velocity2_x", "velocity2_y", "velocity2_z",
                    "angvelocity2_x", "angvelocity2_y", "angvelocity2_z", "drone2_pose",
                    "drone2_vel", 1)

    while not rospy.is_shutdown():

        pass
if __name__ == '__main__':
    main()