import rospy
import math
from geometry_msgs.msg import Quaternion
import numpy as np
from PID_controller import PIDController
inputs = Quaternion()
import  random
class random_policy(object):
	
	def __init__(self):
		# self.epi_num = episode_num
		self.p_freq = 100
		# self.current = np.array(current_xyz)
		# self.desired = np.array(desired_xyz)
		self.inputs = Quaternion()
		self.low_val= [-1,-1,-1,-1]
		self.low_val = [num * 0.05 for num in self.low_val]
		self.high_val= [1,1,1,1]
		self.high_val = [num * 0.05 for num in self.high_val]
		self.shape = [4]
		self.rp_count = 0
		self.PID_z = PIDController(0.2, 0, 0.01)
		self.PID_pitch = PIDController(0.2, 0, 0.01)
		self.PID_roll = PIDController(0.2, 0, 0.01)
		self.PID_yaw = PIDController(0, 0, 0)
		self.epsilon = 0.1

	def P_controller(self):
		current = current_xyz
		desired = desired_xyz

		inputs.w = 0.1 * (desired.z - current.z) / desired.z
		inputs.y = 0
		temp = 180 * (math.atan((desired.y - current.y) / (desired.x - current.x))) / 3.142
		inputs.z = -0.1 * (current.yaw - temp) / 180
		if abs(temp - current.yaw) < 5:
			inputs.x = 0.0001 * math.sqrt((desired.y - current.y)**2 + (desired.x - current.x)**2)
		else:
			inputs.x = 0
		return inputs

	def random_policy(self):
		temp_inputs = np.random.uniform(self.low_val, self.high_val, self.shape)
		in_p = self.P_controller()
		in_noise = Quaternion(x = temp_inputs[0], y =temp_inputs[1], z = temp_inputs[2], w = temp_inputs[3])
		inputs = Quaternion(in_p.x + in_noise.x, in_p.y + in_noise.y, in_p.z + in_noise.z, in_p.w + in_noise.w)
		self.rp_count = self.rp_count + 1
		return inputs

	def PIDcontrol(self, current, desired):

		z_rotation_angle = -current.orientation.z
		# print("Yaw: " + str(z_rotation_angle))

		x_rotated_curr = current.position.x*np.cos(np.radians(z_rotation_angle)) - current.position.y*np.sin(np.radians(z_rotation_angle))
		y_rotated_curr = current.position.x* np.sin(np.radians(z_rotation_angle)) + current.position.y * np.cos(np.radians(z_rotation_angle))

		x_rotated_des = desired.x * np.cos(np.radians(z_rotation_angle)) - desired.y * np.sin(
			np.radians(z_rotation_angle))
		y_rotated_des = desired.x * np.sin(np.radians(z_rotation_angle)) + desired.y * np.cos(
			np.radians(z_rotation_angle))
		# print("Desired X: " + str(y_rotated_des))
		# print('Current X:' + str(y_rotated_curr))

		# desired_roll = math.degrees(math.atan2((y_rotated_des - y_rotated_curr), (desired.z - current.z)))
		# desired_pitch = math.degrees(math.atan2((x_rotated_des - x_rotated_curr), (desired.z - current.z)))

		# desired_roll = math.degrees(math.atan2((desired.y - current.y), (desired.z - current.z)))
		# desired_pitch = math.degrees(math.atan2((desired.x - current.x), (desired.z - current.z)))
		# print(desired_pitch)
		# print('CP:' + str(current.pitch))
		# inputs.w = 0
		inputs.w = self.PID_z.compute(desired.z, current.position.z, 2866) #norm is the heght of the bounds
		# y = self.PID_pitch.compute(desired_pitch, current.pitch, 90
		# inputs.y = np.cos(np.radians(z_rotation_angle))*y
		# inputs.y = 0
		# inputs.y = y
		inputs.y = self.PID_pitch.compute(x_rotated_des, x_rotated_curr,10000)
		# temp = 180 * (math.atan2((desired.y - current.y), (desired.x - current.x))) / 3.142
		desired_yaw = math.degrees(math.atan2((y_rotated_des - y_rotated_curr), (x_rotated_des - x_rotated_curr)))

		# print(temp)
		# inputs.z = 0.2 * (current.yaw) / 180
		yaw_trigger = random.uniform(0,10)
		if yaw_trigger > 2:
			inputs.z = 0
		else:
			inputs.z = random.uniform(-0.25,0.25)
		# inputs.z = self.PID_yaw.compute(desired_yaw, current.yaw, 180)
		# inputs.x = self.PID_roll.compute(desired_roll, current.roll, 90)
		# inputs.x = 0
		inputs.x = self.PID_roll.compute(y_rotated_des,y_rotated_curr, 10000)
		action = [inputs.x, inputs.y, inputs.z, inputs.w]
		return action, inputs

	def exploration_policy(self, current, desired):
		# if (self.epi_num % self.p_freq == 0):
		# 	# return self.random_policy()
		# 	# return self.PIDcontrol()
		# 	return self.PIDcontrol()
		#
		# else:
		self.rp_count +=1

		min_action = -1
		max_action = 1
		inputs, action = self.PIDcontrol(current, desired)
		# if self.rp_count > 1000:
		# 	action.x = 0
		# 	action.y = 0
		# 	action.z = 0
		# 	action.w = 0
		# 	inputs = [action.x, action.y, action.z, action.w]
		# else:
		inputs = np.clip(inputs, min_action, max_action)
		action.x = max(min(action.x, max_action), min_action)
		action.y = max(min(action.y, max_action), min_action)
		action.z = max(min(action.z, max_action), min_action)
		action.w = max(min(action.w, max_action), min_action)
		return inputs, action
		

	def epsilon_greedy(self):
		if random.random() < self.epsilon:
			# Explore: Choose a random action
			inputs.x = (random.randint(-3, 3))/20
			inputs.y= (random.randint(-3, 3))/20
			inputs.z = 0
			inputs.w = (random.randint(-3, 3))/20
			return inputs
		else:
			return self.PIDcontrol()
