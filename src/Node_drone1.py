#!/usr/bin/env python3
from drone_pubsub import *

def main():
    print("Drone1 Node")
    rospy.init_node('drone1_node')
    drone1 = PubSub("current_x", "current_y", "current_z", "roll", "pitch",
                    "yaw", "velocity_x", "velocity_y", "velocity_z",
                    "angvelocity_x", "angvelocity_y", "angvelocity_z", "drone1_pose",
                    "drone1_vel", 1)

    while not rospy.is_shutdown():

        pass
if __name__ == '__main__':
    main()