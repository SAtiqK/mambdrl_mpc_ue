#!/usr/bin/env python3
import math
import time

# def to_print():

# if __name__ == '__main__':
# 	print('Hello World')

import rospy
from pubsub_synced import *
import tensorflow as tf
import os
from training import train
import numpy as np
import yaml
import argparse
import json
from random_policy import random_policy
from numpy import linalg as LA

# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt

from PID_controller import PIDController
from data_save import data_save
from visualization import visualize_rollout


expl_policy = random_policy()

# episode_counter
# steps_rollout_counter

def sub6_callback(self, msg):
	# episode_number = episode_number + 1
	# print('Inside callback')
	if self.flagdz == True:
		value = msg.data
		y_f = float(value)
		desired_xyz.z = y_f
		begin = time.time()
		# print("callbackFINALSTART")
		self.flagdz = False


	# print(current_xyz.z)


	#add the random policy rollout here
	#check if

	#needs to check for terminal state c
	#needs a counter for the number of steps in a rollout
	#needs a counter for the number of episodes
	# self.pause.publish(True)

	# if (drone_status.inbound == False):
	# 	print('Out of Bound')
	# 	# steps_count = 0
	# if (drone_status.crashed == True):
	# 	print('Crashed')
		# steps_count = 0

		if vis == True:
			p, o = vis_obj.visualize(self)
			pass
		else:
			inputs = train(self, current_xyz, desired_xyz, current_vel, drone_status.crashed, drone_status.inbound, epi_info,
					   traindata, expl_policy)
			# inputs.w = 0
			# inputs.x = 0
			# inputs.y = 1
			# inputs.z = 0
			end = time.time()
			# print(LA.norm([current_vel.x, current_vel.y, current_vel.z]))
			self.pause.publish(False)
			self.chatter_pub.publish(inputs)
			time_action = end - begin
			# sleep_time = (1/60) - time_action
			sleep_time = 1 / 60
			# if sleep_time < 0:
			# 	sleep_time = 0
			rospy.sleep(sleep_time)
			self.pause.publish(True)
		self.flags_reset()

	# print("callbackFINALEND")
	# inputs = mpc_control.get_action(current_xyz, desired_xyz)
	# inputs.y = 1




def main():
	global steps_count
	steps_count = 0
	yaml_path = os.path.abspath('UEdrone_train.yaml')
	assert (os.path.exists(yaml_path))
	print("no assertion error")
	with open(yaml_path, 'r') as f:
		params = yaml.safe_load(f)  # changed from yaml.load() to yaml.saf_load()
	get_new_data = params['flags']['get_new_data']
	global vis
	vis = params['flags']['only_visualize']


	parser = argparse.ArgumentParser()
	# the yaml file that has all the params required
	parser.add_argument('--run_num', type=int, default=300)
	args = parser.parse_args()
	global RUN_NUMBER
	RUN_NUMBER = args.run_num
	save_dir = 'run_' + str(RUN_NUMBER)
	global traindata
	traindata = data_save(RUN_NUMBER)
	counter_agg = 2
	if vis == True:
		with open(save_dir + '/training_data/resulting_x4.json', 'r') as json_file:
			# Load the JSON data into a Python data structure
			states_for_vis = json.load(json_file)
		global  vis_obj
		vis_obj = visualize_rollout(states_for_vis, counter_agg, save_dir, len(states_for_vis))


	# make an object for the dynamics model
	# make an object for the MPC controller
	train_dyn_model = False #add these in the callback functions
	mpc_rollout = False #to check if the rollout is based on an MPC or is a random policy
	episode_counter = 1
	steps_rollout_counter = 0
	steps = 0
	horizon = 30

	num_control_samples = 500
	steps_per_episode = 1500
	fraction_use_new = 0
	gpu_devices = tf.config.experimental.list_physical_devices('GPU')
	for device in gpu_devices:
		tf.config.experimental.set_memory_growth(device, True)
	gpu_device = 0
	gpu_frac = 0.9
	tf_datatype = tf.float64
	print_minimal = True
	global Z_saved
	Z_saved = []

	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
	gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
	config = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
									  log_device_placement=False,
									  allow_soft_placement=True,
									  inter_op_parallelism_threads=1,
									  intra_op_parallelism_threads=1)
		# rospy.init_node("training_data")
		# PubSub.sub6callback = sub6_callback
		#
		# controller = PubSub("chatter", "steps","current_x", "current_y", "current_z", "desired_location", "desired_y", "desired_z", "current_yaw", "crashed", "inbound", 'roll', 'pitch', 'sim_satus', 'pause', 1)
	if not (get_new_data):
		dataX = np.load(save_dir + '/training_data/dataX.npy')  # input1: state
		dataY = np.load(save_dir + '/training_data/dataY.npy')  # input2: control
		dataZ = np.load(save_dir + '/training_data/dataZ.npy')
	elif vis == False:
		get_init_data = True
		rospy.init_node("training_data")
		PubSub.sub6callback = sub6_callback

		controller = PubSub("chatter", "steps", "current_x", "current_y", "current_z", "desired_location",
							"desired_y", "desired_z", "current_yaw", "crashed", "inbound", 'roll', 'pitch',
							'sim_satus', 'pause','node_status', "velocity_x", "velocity_y", "velocity_z",
							"angvelocity_x", "angvelocity_y", "angvelocity_z",1)
		print("Got here")

		while not rospy.is_shutdown():
			# print('before training')
			# inputs = train(controller, current_xyz, desired_xyz, drone_status.crashed, drone_status.inbound, epi_info)
			# controller.chatter_pub.publish(inputs)
			# print('after pub')

			# if train_dyn_model:
			#     training_loss, old_loss, new_loss = dyn_model.train(inputs, outputs, inputs_new, outputs_new, nEpoch, save_dir, fraction_use_new)
			# else:
			#     pass
			# print('Here')
			pass
		# dataX = np.load(save_dir + '/training_data/dataX.npy')  # input1: state
		# dataY = np.load(save_dir + '/training_data/dataY.npy')  # input2: control
		# dataZ = np.load(save_dir + '/training_data/dataZ.npy')  # output

		rospy.signal_shutdown("Done getting data")
	else:
		rospy.init_node('Visualization')
		vis_pose = Vis_pub("vis_pos", "vis_orient", "des_wp", 1)

		while not rospy.is_shutdown():
			p1, o1 = vis_obj.visualize(vis_pose)
			if p1 == 100 :
				pass
			else:
				p = Vector3(p1[0], p1[1], p1[2])
				o = Vector3(o1[0], o1[1], o1[2])
				vis_pose.position.publish(p)
				vis_pose.orientation.publish(o)
				rospy.sleep(1/60)


if __name__ == "__main__":
	try:
		main()
	except rospy.ROSInterruptException:
		pass

