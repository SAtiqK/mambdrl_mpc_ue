import sys

import rospy
from location_ue import uelocation
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from flags import drone_flags
from episode_info import episodeInfo



current_xyz = uelocation()
desired_xyz = uelocation()
current_vel = uelocation()
drone_status = drone_flags()
epi_info = episodeInfo()
# yaw = float(0)
class PubSub:

    def __init__(self, publish_topic_name, stepF, sub1_t, sub2_t, sub3_t, sub4_t, sub5_t, sub6_t, sub7_t, crashed,
                 inbound, roll, pitch, ss, pause, node_ss, sub10_t, sub11_t, sub12_t, sub13_t, sub14_t,
                 sub15_t, queue_size):
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
        self.flagdx = False
        self.flagdy = False
        self.flagdz = False
        self.sims = rospy.Subscriber(ss, String, self.ssCallBack, queue_size= queue_size)
        self.chatter_pub = rospy.Publisher(publish_topic_name, Quaternion, queue_size=queue_size)
        self.steps = rospy.Publisher(stepF, Bool, queue_size= queue_size)
        self.pause = rospy.Publisher(pause, Bool, queue_size = queue_size)
        self.node_status = rospy.Publisher(node_ss, Bool,  queue_size = queue_size)
        self.sub7 = rospy.Subscriber(sub7_t, String, self.sub7callback, queue_size=queue_size)
        self.sub1 = rospy.Subscriber(sub1_t, String, self.sub1callback, queue_size=queue_size)
        self.sub2 = rospy.Subscriber(sub2_t, String, self.sub2callback, queue_size=queue_size)
        self.sub3 = rospy.Subscriber(sub3_t, String, self.sub3callback, queue_size=queue_size)
        self.sub4 = rospy.Subscriber(sub4_t, String, self.sub4callback, queue_size=queue_size)
        self.sub5 = rospy.Subscriber(sub5_t, String, self.sub5callback, queue_size=queue_size)
        self.subroll = rospy.Subscriber(roll, String, self.rollcallback, queue_size=queue_size)
        self.subpitch = rospy.Subscriber(pitch, String, self.pitchcallback, queue_size=queue_size)
        self.sub6 = rospy.Subscriber(sub6_t, String, self.sub6callback, queue_size=queue_size)
        self.crashed = rospy.Subscriber(crashed, String, self.crashedCallBack, queue_size = queue_size)
        self.inbound = rospy.Subscriber(inbound, String, self.inboundCallBack, queue_size = queue_size)
        self.sub10 = rospy.Subscriber(sub10_t, String, self.velocityx_callback, queue_size=queue_size)
        self.sub11 = rospy.Subscriber(sub11_t, String, self.velocityy_callback, queue_size=queue_size)
        self.sub12 = rospy.Subscriber(sub12_t, String, self.velocityz_callback, queue_size=queue_size)
        self.sub13 = rospy.Subscriber(sub13_t, String, self.angvelocityx_callback, queue_size=queue_size)
        self.sub14 = rospy.Subscriber(sub14_t, String, self.angvelocityy_callback, queue_size=queue_size)
        self.sub15 = rospy.Subscriber(sub15_t, String, self.angvelocityz_callback, queue_size=queue_size)


        print('PUBSUB made')
        # self.nH = rospy.NodeHandle()

    def sub1callback(self, msg):
        value = msg.data
        if self.flagstart == True:
            y_f = float(value)
            current_xyz.x = y_f
            # print("current x: " + str(y_f))
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
            self.flagz = True
            self.flagy = False
        else:
            pass

    def sub4callback(self, msg):
        if self.flagvpitch == True:
            value = msg.data
            y_f = float(value)
            desired_xyz.x = y_f
            # print("desired x: " + str(y_f))
            self.flagvpitch = False
            self.flagdx = True
        else:
            pass

    def sub5callback(self, msg):
        if self.flagdx == True:
            value = msg.data
            y_f = float(value)
            desired_xyz.y = y_f
            # print("desired y:" + str(y_f))
            self.flagdy = True
            self.flagdx = False

    def sub7callback(self, msg):
        value = msg.data
        if self.flagz == True:
            y_f = float(value)
            current_xyz.yaw = y_f
            # print("current yaw:" + str(y_f))
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
        value = msg.data
        drone_status.inbound = bool(int(value))

    def ssCallBack(self, msg):
        value = msg.data
        epi_info.simu_status = bool(int(value))

    def rollcallback(self, msg):
        if self.flagpitch == True:
            value = msg.data
            current_xyz.roll = float(value)
            # print("current roll: " + str(value))
            self.flagroll= True
            self.flagpitch = False
        else:
            pass
    def pitchcallback(self, msg):
        if self.flagyaw== True:
            value = msg.data
            current_xyz.pitch = float(value)
            # print("current pitch:" + str(value))
            self.flagpitch = True
            self.flagyaw = False
        else:
            pass

    def velocityx_callback(self, msg):
        if self.flagroll == True:
            value = msg.data
            current_vel.x = float(value)
            self.flagvx = True
            self.flagroll = False
        else:
            pass

    def velocityy_callback(self, msg):
        if self.flagvx == True:
            value = msg.data
            current_vel.y = float(value)
            self.flagvy = True
            self.flagvx = False
        else:
            pass

    def velocityz_callback(self, msg):
        if self.flagvy == True:
            value = msg.data
            current_vel.z = float(value)
            self.flagvz = True
            self.flagvy = False
        else:
            pass

    def angvelocityx_callback(self, msg):
        if self.flagvz == True:
            value = msg.data
            current_vel.pitch = float(value)
            self.flagvyaw = True
            self.flagvz = False
        else:
            pass

    def angvelocityy_callback(self, msg):
        if self.flagvyaw == True:
            value = msg.data
            current_vel.roll = float(value)
            self.flagvroll = True
            self.flagvyaw = False

    def angvelocityz_callback(self, msg):
        if self.flagvroll == True:
            value = msg.data
            current_vel.yaw = float(value)
            self.flagvpitch = True
            self.flagvroll = False
        # print("being called")

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
        self.desired_waypoint = rospy.Publisher(desired, Vector3, queue_size = queue_size)
