#!/usr/bin/env python3
from drone_pubsub import *

def main():
    print("Drone4 Node")
    rospy.init_node('drone4_node')
    drone4 = PubSub("current4_x", "current4_y", "current4_z", "roll4", "pitch4",
                    "yaw4", "velocity4_x", "velocity4_y", "velocity4_z",
                    "angvelocity4_x", "angvelocity4_y", "angvelocity4_z", "drone4_pose",
                    "drone4_vel", 1)

    while not rospy.is_shutdown():

        pass
if __name__ == '__main__':
    main()