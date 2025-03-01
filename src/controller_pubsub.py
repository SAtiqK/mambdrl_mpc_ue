#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from std_msgs.msg import String
from geometry_msgs.msg import Quaternion

from flags import drone_flags


class PubSub:

    def __init__(self, sub1_t, sub2_t, sub3_t, sub4_t, sub5_t, sub6_t, sub7_t, sub8_t, sub9_t, sub10_t, sub11_t,
                 d1_inp, d2_inp, d3_inp, d4_inp, crashed, inbound, ss, steps_topic, pause_topic, node_ss, queue_size):
        self.drone1_pose = Pose()
        self.drone1_vel = Pose()
        self.drone2_pose = Pose()
        self.drone2_vel = Pose()
        self.drone3_pose = Pose()
        self.drone3_vel = Pose()
        self.drone4_pose = Pose()
        self.drone4_vel = Pose()
        self.payload_pose = Pose()
        self.payload_vel = Pose()
        self.desired = Vector3()
        self.drone_status = drone_flags()

        self.flagdrone1 = False
        self.flagdrone2 = False
        self.flagdrone3 = False
        self.flagdrone4 = False
        self.flagdrone1_vel = False
        self.flagdrone2_vel = False
        self.flagdrone3_vel = False
        self.flagdrone4_vel = False
        self.flagpayload = False
        self.flagpayload_vel = False
        self.flagdesired = False
        self.flagstart = True

        self.drone_sub = rospy.Subscriber(sub1_t, Pose, self.sub1callback, queue_size=queue_size)
        self.drone2_sub = rospy.Subscriber(sub2_t, Pose, self.sub2callback, queue_size=queue_size)
        self.drone3_sub = rospy.Subscriber(sub3_t, Pose, self.sub3callback, queue_size=queue_size)
        self.drone4_sub = rospy.Subscriber(sub4_t, Pose, self.sub4callback, queue_size=queue_size)
        self.drone1_vel_sub = rospy.Subscriber(sub5_t, Pose, self.sub5callback, queue_size=queue_size)
        self.drone2_vel_sub = rospy.Subscriber(sub6_t, Pose, self.sub6callback, queue_size=queue_size)
        self.drone3_vel_sub = rospy.Subscriber(sub7_t, Pose, self.sub7callback, queue_size=queue_size)
        self.drone4_vel_sub = rospy.Subscriber(sub8_t, Pose, self.sub8callback, queue_size=queue_size)
        self.payload_sub = rospy.Subscriber(sub9_t, Pose, self.sub9callback, queue_size=queue_size)
        self.payload_vel_sub = rospy.Subscriber(sub10_t, Pose, self.sub10callback, queue_size=queue_size)
        self.desired_position = rospy.Subscriber(sub11_t, Vector3, self.sub11callback, queue_size=queue_size)
        self.inputs_drone1 = rospy.Publisher(d1_inp, Quaternion, queue_size=queue_size)
        self.inputs_drone2 = rospy.Publisher(d2_inp, Quaternion, queue_size=queue_size)
        self.inputs_drone3 = rospy.Publisher(d3_inp, Quaternion, queue_size=queue_size)
        self.inputs_drone4 = rospy.Publisher(d4_inp, Quaternion, queue_size=queue_size)
        self.steps = rospy.Publisher(steps_topic, Bool, queue_size=queue_size)
        self.pause = rospy.Publisher(pause_topic, Bool, queue_size=queue_size)
        self.node_status = rospy.Publisher(node_ss, Bool, queue_size=queue_size)
        self.sims = rospy.Subscriber(ss, String, self.ssCallBack, queue_size=queue_size)
        self.crashed = rospy.Subscriber(crashed, String, self.crashedCallBack, queue_size=queue_size)
        self.inbound = rospy.Subscriber(inbound, String, self.inboundCallBack, queue_size=queue_size)

        print('Controller Node Created')
        # self.nH = rospy.NodeHandle()

    def sub1callback(self, msg):
        if self.flagstart == True:
            y_f = msg
            self.drone1_pose = y_f
            self.flagdrone1 = True
            self.flagstart = False
        else:
            pass

    def sub2callback(self, msg):
        if self.flagdrone1 == True:
            y_f = msg
            self.drone2_pose = y_f
            self.flagdrone2 = True
            self.flagdrone1 = False
        else:
            pass

    def sub3callback(self, msg):
        if self.flagdrone2 == True:
            y_f = msg
            self.drone3_pose = y_f
            self.flagdrone3 = True
            self.flagdrone2 = False
        else:
            pass

    def sub4callback(self, msg):
        if self.flagdrone3 == True:
            y_f = msg
            self.drone4_pose = y_f
            self.flagdrone4 = True
            self.flagdrone3 = False
        else:
            pass

    def sub5callback(self, msg):
        if self.flagdrone4 == True:
            y_f = msg
            self.drone1_vel= y_f
            self.flagdrone1_vel = True
            self.flagdrone4 = False
        else:
            pass

    def sub6callback(self, msg):
        if self.flagdrone1_vel == True:
            y_f = msg
            self.drone2_vel= y_f
            self.flagdrone1_vel = False
            self.flagdrone2_vel = True
        else:
            pass

    def sub7callback(self, msg):
        if self.flagdrone2_vel == True:
            y_f = msg
            self.drone3_vel = y_f
            self.flagdrone3_vel = True
            self.flagdrone2_vel = False
        else:
            pass

    def sub8callback(self, msg):
        if self.flagdrone3_vel == True:
            y_f = msg
            self.drone4_vel = y_f
            self.flagdrone4_vel = True
            self.flagdrone3_vel = False
        else:
            pass

    def sub9callback(self, msg):
        if self.flagdrone4_vel == True:
            y_f = msg
            self.payload_pose = y_f
            self.flagpayload = True
            self.flagdrone4_vel = False
        else:
            pass

    def sub10callback(self, msg):
        if self.flagpayload == True:
            y_f = msg
            self.payload_vel = y_f
            self.flagpayload_vel= True
            self.flagpayload = False
        else:
            pass

    # def sub11callback(self, msg):
    #     if self.flagpayload_vel == True:
    #         y_f = msg
    #         self.payload_pose = y_f
    #         self.flagpayload_vel= False
    #         self.flagdesired = True
    #         self.flags_reset()
    #     else:
    #         pass

    def crashedCallBack(self, msg):
        value = msg.data
        c = bool(int(value))
        self.drone_status.crashed = c

    def inboundCallBack(self, msg):
        value = msg.data
        self.drone_status.inbound = bool(int(value))

    def ssCallBack(self, msg):
        value = msg.data
        self.drone_status.sim = bool(int(value))

    def flags_reset(self):
        self.flagdrone1 = False
        self.flagdrone2 = False
        self.flagdrone3 = False
        self.flagdrone4 = False
        self.flagdrone1_vel = False
        self.flagdrone2_vel = False
        self.flagdrone3_vel = False
        self.flagdrone4_vel = False
        self.flagpayload = False
        self.flagpayload_vel = False
        self.flagdesired = False
        self.flagstart = True

class Vis_pub:
    def __init__(self, position, orientation, desired, queue_size):
        self.position = rospy.Publisher(position, Vector3, queue_size=queue_size)
        self.orientation = rospy.Publisher(orientation, Vector3, queue_size=queue_size)
        self.desired_waypoint = rospy.Publisher(desired, Vector3, queue_size=queue_size)

