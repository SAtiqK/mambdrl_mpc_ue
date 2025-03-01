#!/usr/bin/env python3
from drone_pubsub import *

def main():
    print("Drone3 Node")
    rospy.init_node('drone3_node')
    drone3 = PubSub("current3_x", "current3_y", "current3_z", "roll3", "pitch3",
                    "yaw3", "velocity3_x", "velocity3_y", "velocity3_z",
                    "angvelocity3_x", "angvelocity3_y", "angvelocity3_z", "drone3_pose",
                    "drone3_vel", 1)

    while not rospy.is_shutdown():

        pass
if __name__ == '__main__':
    main()