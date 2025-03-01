#!/usr/bin/env python3
from payload_pubsub import *

def main():
    print("Payload Node")
    rospy.init_node('pl_node')
    pl = PubSub("pl_current_x", "pl_current_y", "pl_current_z", "pl_roll", "pl_pitch",
                    "pl_yaw", "pl_velocity_x", "pl_velocity_y", "pl_velocity_z",
                    "pl_angvelocity_x", "pl_angvelocity_y", "pl_angvelocity_z", "pl_pose",
                    "pl_vel", "desired_location", "desired_y", "desired_z", "desired_wp",1)

    while not rospy.is_shutdown():

        pass
if __name__ == '__main__':
    main()