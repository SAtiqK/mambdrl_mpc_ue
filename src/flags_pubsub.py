#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

class PubSub:

    def __init__(self, crashed, inbound, ss, steps_topic, pause_topic, node_ss, queue_size):

        self.sims = rospy.Subscriber(ss, String, self.ssCallBack, queue_size=queue_size)
        self.crashed = rospy.Subscriber(crashed, String, self.crashedCallBack, queue_size=queue_size)
        self.inbound = rospy.Subscriber(inbound, String, self.inboundCallBack, queue_size=queue_size)
        self.sims = rospy.Subscriber(ss, String, self.ssCallBack, queue_size=queue_size)
        self.crashed = rospy.Subscriber(crashed, String, self.crashedCallBack, queue_size=queue_size)
        self.inbound = rospy.Subscriber(inbound, String, self.inboundCallBack, queue_size=queue_size)


        print('PUBSUB made')
        # self.nH = rospy.NodeHandle()

    def sub1callback(self, msg):
        value = msg.data
        if self.flagstart == True:
            y_f = float(value)
            current_xyz.x = y_f
            # print("current x: " + str(y_f))
            # print("1")
            self.flagx = True
            self.flagstart = False
        else:
            pass

    def sub2callback(self, msg):
        value = msg.data
        if self.flagx == True:
            y_f = float(value)
            current_xyz.y = y_f
            # print("current y: " + str(y_f))
            # print("2")
            self.flagy = True
            self.flagx = False
        else:
            pass

    def sub3callback(self, msg):
        value = msg.data
        if self.flagy == True:
            y_f = float(value)
            current_xyz.z = y_f
            # print("current z: " + str(y_f))
            # print("3")
            self.flagz = True
            self.flagy = False
        else:
            pass

    def sub4callback(self, msg):
        if self.flag_plvelz == True:
            value = msg.data
            y_f = float(value)
            desired_xyz.x = y_f
            # print("desired x: " + str(y_f))
            # print("25")
            self.flag_yaw = False
            self.flagdx= True
        else:
            pass

    def sub5callback(self, msg):
        if self.flagdx == True:
            value = msg.data
            y_f = float(value)
            desired_xyz.y = y_f
            # print("26")
            # print("desired y:" + str(y_f))
            self.flagdy = True
            self.flagdx = False

    def sub7callback(self, msg):
        value = msg.data
        if self.flagz == True:
            y_f = float(value)
            current_xyz.yaw = y_f
            # print("current yaw:" + str(y_f))
            # print("4")
            self.flagyaw = True
            self.flagz = False
        else:
            pass
        # print(yaw)

    def crashedCallBack(self, msg):
        value = msg.data
        c = bool(int(value))
        drone_status.crashed = c

    def inboundCallBack(self, msg):
        value = msg.data3
        drone_status.inbound = bool(int(value))

    def ssCallBack(self, msg):
        value = msg.data
        epi_info.simu_status = bool(int(value))


