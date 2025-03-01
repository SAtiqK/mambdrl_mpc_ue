#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Pose

class PubSub:

    def __init__(self, current_x, current_y,current_z, current_roll, current_pitch, current_yaw, current_vel_x, current_vel_y,
                 current_vel_z, current_vel_roll, current_vel_pitch, current_vel_yaw, pub_pose, pub_vel, desired_x,
                 desired_y, desired_z, desiredwp, queue_size):

        self.pl_pose = Pose()
        self.pl_vel = Pose()
        self.desired_wp = Vector3()

        self.flagx = False
        self.flagy = False
        self.flagz = False
        self.flagyaw = False
        self.flagroll = False
        self.flagpitch = False
        self.flagvx = False
        self.flagvy = False
        self.flagvz = False
        self.flagvyaw = False
        self.flagvroll = False
        self.flagvpitch = False
        self.flagdx = False
        self.flagdy = False
        self.flagdz = False
        self.flagstart = True

        self.curr_x = rospy.Subscriber(current_x, String, self.sub1callback, queue_size=queue_size)
        self.curr_y  = rospy.Subscriber(current_y, String, self.sub2callback, queue_size=queue_size)
        self.curr_z  = rospy.Subscriber(current_z, String, self.sub3callback, queue_size=queue_size)
        self.curr_roll  = rospy.Subscriber(current_roll, String, self.sub4callback, queue_size=queue_size)
        self.curr_pitch  = rospy.Subscriber(current_pitch, String, self.sub5callback, queue_size=queue_size)
        self.curr_yaw  = rospy.Subscriber(current_yaw, String, self.sub6callback, queue_size=queue_size)
        self.curr_velx  = rospy.Subscriber(current_vel_x, String, self.sub7callback, queue_size=queue_size)
        self.curr_vely  = rospy.Subscriber(current_vel_y, String, self.sub8callback, queue_size=queue_size)
        self.curr_velz  = rospy.Subscriber(current_vel_z, String, self.sub9callback, queue_size=queue_size)
        self.curr_vroll  = rospy.Subscriber(current_vel_roll, String, self.sub10callback, queue_size=queue_size)
        self.curr_vpitch  = rospy.Subscriber(current_vel_pitch, String, self.sub11callback, queue_size=queue_size)
        self.curr_vyaw  = rospy.Subscriber(current_vel_yaw, String, self.sub12callback, queue_size=queue_size)
        self.des_x = rospy.Subscriber(desired_x, String, self.sub13callback, queue_size=queue_size)
        self.des_y = rospy.Subscriber(desired_y, String, self.sub14callback, queue_size=queue_size)
        self.des_z = rospy.Subscriber(desired_z, String, self.sub15callback, queue_size=queue_size)
        self.curr_pose = rospy.Publisher(pub_pose, Pose, queue_size=queue_size)
        self.curr_vel = rospy.Publisher(pub_vel, Pose, queue_size=queue_size)
        self.des_wp = rospy.Publisher(desiredwp, Vector3, queue_size=queue_size)

        print('PUBSUB made')
        # self.nH = rospy.NodeHandle()

    def sub1callback(self, msg):
        value = msg.data
        if self.flagstart == True:
            y_f = float(value)
            self.pl_pose.position.x = y_f
            self.flagx = True
            self.flagstart = False
        else:
            pass

    def sub2callback(self, msg):
        value = msg.data
        if self.flagx == True:
            y_f = float(value)
            self.pl_pose.position.y = y_f
            self.flagy = True
            self.flagx = False
        else:
            pass

    def sub3callback(self, msg):
        value = msg.data
        if self.flagy == True:
            y_f = float(value)
            self.pl_pose.position.z = y_f
            self.flagz = True
            self.flagy = False
        else:
            pass

    def sub4callback(self, msg):
        if self.flagz== True:
            value = msg.data
            y_f = float(value)
            self.pl_pose.orientation.x = y_f
            self.flagz = False
            self.flagroll= True
        else:
            pass

    def sub5callback(self, msg):
        if self.flagroll == True:
            value = msg.data
            y_f = float(value)
            self.pl_pose.orientation.y = y_f
            self.flagpitch = True
            self.flagroll = False

    def sub6callback(self, msg):
        value = msg.data
        if self.flagpitch == True:
            y_f = float(value)
            self.pl_pose.orientation.z = y_f
            self.flagyaw = True
            self.flagpitch = False

    def sub7callback(self, msg):
        if self.flagyaw == True:
            value = msg.data
            self.pl_vel.position.x = float(value)
            # print("7")
            self.flagvx = True
            self.flagyaw = False
        else:
            pass

    def sub8callback(self, msg):
        if self.flagvx == True:
            value = msg.data
            self.pl_vel.position.y = float(value)
            # print("8")
            self.flagvy = True
            self.flagvx = False
        else:
            pass

    def sub9callback(self, msg):
        if self.flagvy == True:
            value = msg.data
            self.pl_vel.position.z  = float(value)
            # print("9")
            self.flagvz = True
            self.flagvy = False
        else:
            pass

    def sub10callback(self, msg):
        if self.flagvz == True:
            value = msg.data
            self.pl_vel.orientation.x = float(value)
            # print("10")
            self.flagvroll = True
            self.flagvz = False
        else:
            pass

    def sub11callback(self, msg):
        if self.flagvroll== True:
            value = msg.data
            self.pl_vel.orientation.y = float(value)
            # print("11")
            self.flagvpitch = True
            self.flagvroll = False

    def sub12callback(self, msg):
        if self.flagvpitch== True:
            value = msg.data
            self.pl_vel.orientation.z = float(value)
            self.flagvyaw = True
            self.flagvpitch = False

    def sub13callback(self, msg):
        if self.flagvyaw == True:
            value = msg.data
            self.desired_wp.x = float(value)
            self.flagdx = True
            self.flagvyaw = False

    def sub14callback(self, msg):
        if self.flagdx == True:
            value = msg.data
            self.desired_wp.y = float(value)
            self.flagdy = True
            self.flagdx= False

    def sub15callback(self, msg):
        if self.flagdy == True:
            value = msg.data
            self.desired_wp.z = float(value)
            self.flagdz = True
            self.flagdy = False
            self.flags_reset()

            self.curr_pose.publish(self.pl_pose)
            self.curr_vel.publish(self.pl_vel)
            self.des_wp.publish(self.desired_wp)

    def flags_reset(self):
        self.flagx = False
        self.flagy = False
        self.flagz = False
        self.flagyaw = False
        self.flagroll = False
        self.flagpitch = False
        self.flagvx = False
        self.flagvy = False
        self.flagvz = False
        self.flagvyaw = False
        self.flagvroll = False
        self.flagvpitch = False
        self.flagstart = True


class Vis_pub:
    def __init__(self, position, orientation, desired, queue_size):
        self.position = rospy.Publisher(position, Vector3, queue_size=queue_size)
        self.orientation = rospy.Publisher(orientation, Vector3, queue_size=queue_size)
        self.desired_waypoint = rospy.Publisher(desired, Vector3, queue_size=queue_size)
