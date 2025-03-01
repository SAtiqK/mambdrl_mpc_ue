#!/usr/bin/env python3
import math
import pathlib
from controller_pubsub import *
import argparse
import tensorflow as tf
import os
import yaml
import numpy as np
import json
import torch
from training import train
from dynamics_model import Dyn_Model
from mpc_controller_v2 import MPCController
from PETS_model import nn_constructor
from dotmap import DotMap

status = True


def sub11_callback(self, msg):
	if self.flagpayload_vel == True:
		global status
		status = False
		self.desired = msg
		self.flagpayload_vel = False
		self.flagdesired = True
		self.flags_reset()


	# mpc_control.to_check()
		if Multi_Agent:
			drones_pose = [self.drone1_pose, self.drone2_pose, self.drone3_pose, self.drone4_pose]
			drones_vel = [self.drone1_vel, self.drone2_vel, self.drone3_vel, self.drone4_vel]
			if PID:
				inputs = train(self, drones_pose, self.desired, drones_vel, self.payload_pose,
						self.payload_vel, self.drone_status.crashed, self.drone_status.inbound, epi_info,
					   traindata, expl_policy, Multi_Agent)
			else:
				# desired_position = [self.desired.x, self.desired.y, self.desired.z]
				inputs = mpc_control.rollout_step(self, drones_pose, self.desired, drones_vel, self.payload_pose,
												  self.payload_vel, self.drone_status.inbound, self.drone_status.crashed,
												  save_dir, counter_agg, restored_model, PETS, Multi_Agent, drones_pose)
			# self.pause.publish(False)
			self.inputs_drone1.publish(inputs[0])
			self.inputs_drone2.publish(inputs[1])
			self.inputs_drone3.publish(inputs[2])
			self.inputs_drone4.publish(inputs[3])
			# rospy.sleep(1/10)
			# self.pause.publish(True)
		else:
			combined_state = [self.drone1_pose, self.drone2_pose, self.drone3_pose, self.drone4_pose]

			inputs = mpc_control.rollout_step(self, self.drone1_pose, self.desired, self.drone1_vel, self.payload_pose,
											  self.payload_vel, self.drone_status.inbound, self.drone_status.crashed,
										  save_dir, counter_agg, restored_model, PETS, Multi_Agent, combined_state)

			# self.pause.publish(False)
			self.inputs_drone1.publish(inputs[0])
			self.inputs_drone2.publish(inputs[1])
			self.inputs_drone3.publish(inputs[2])
			self.inputs_drone4.publish(inputs[3])
			# rospy.sleep(1/10)
			# self.pause.publish(True)

def main():
	print("here")

	parser = argparse.ArgumentParser()
	# the yaml file that has all the params required
	parser.add_argument('--run_num', type=int, default=5006)
	parser.add_argument('--yaml_path', type=str,
						default='/UEdrone_train.yaml')
	parser.add_argument('--counter_agg', type=int, default=2)

	args = parser.parse_args()
	global counter_agg
	counter_agg = args.counter_agg
	run_num = args.run_num
	global save_dir
	save_dir = "run_" + str(run_num)
	yaml_path = str(pathlib.Path().resolve()) + args.yaml_path
	with open(yaml_path, 'r') as f:
		params = yaml.safe_load(f)

	# Dynamic Model parameters
	use_PETS = params['flags']['use_PETS']
	lr = params['dyn_model']['lr']
	batchsize = params['dyn_model']['batchsize']
	num_fc_layers = params['dyn_model']['num_fc_layers']
	depth_fc_layers = params['dyn_model']['depth_fc_layers']
	tf_datatype = tf.float64
	global PETS
	PETS = use_PETS
	global Multi_Agent
	Multi_Agent = False
	global PID
	PID = False

	if PID:
		from random_policy import random_policy
		from data_save import data_save
		from episode_info import episodeInfo
		global expl_policy
		expl_policy = random_policy()
		global traindata, epi_info
		traindata = data_save(run_num)
		epi_info = episodeInfo()
	else:
	# MPC parameters
		horizon = params['controller']['horizon']
		num_control_samples = params['controller']['num_control_samples']
		steps_per_episode = params['steps']['steps_per_episode']
		num_trajectories_for_aggregation = params['aggregation']['num_trajectories_for_aggregation']
		num_aggreg = params['aggregation']['num_aggregation_iters']

		print_minimal = params['flags']['print_minimal']

		with open(save_dir + "/training_data/data_stats.json", "r") as json_file:
			# Load the JSON data into a Python dictionary
			data_info = json.load(json_file)

		inputSize = data_info["inputSize"]
		outputSize = data_info["outputSize"]
		mean_x = np.array(data_info["mean_x"])
		mean_y = np.array(data_info["mean_y"])
		mean_z = np.array(data_info["mean_z"])
		std_x = data_info["std_x"]
		std_y = data_info["std_y"]
		std_z = data_info["std_z"]

		global mpc_control

	gpu_devices = tf.config.experimental.list_physical_devices('GPU')
	for device in gpu_devices:
		tf.config.experimental.set_memory_growth(device, True)
	gpu_device = 0
	gpu_frac = 0.9

	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
	gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
	config = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
									  log_device_placement=False,
									  allow_soft_placement=True,
									  inter_op_parallelism_threads=1,
									  intra_op_parallelism_threads=1)

	gpu_devices = tf.config.experimental.list_physical_devices('GPU')
	for device in gpu_devices:
		tf.config.experimental.set_memory_growth(device, True)

	# with tf.compat.v1.Session(config=config) as sess:

	# sess.run(tf.compat.v1.global_variables_initializer())
	global restored_model

	if not PID:
		if use_PETS:
			model_init_config = DotMap(num_nets = 3, model_in = inputSize, model_out = outputSize)
			dyn_model = nn_constructor(model_init_config)
			checkpoint = torch.load(save_dir + '/models/finalModel.pth')

			#
			restored_model = dyn_model.load_state_dict(checkpoint)
		else:
			dyn_model = Dyn_Model(inputSize, outputSize, 1, lr, batchsize, num_fc_layers,
								  depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype,
								  print_minimal=False)
			restored_model = tf.keras.models.load_model(save_dir + "/models/finalModel7.keras")

		global mpc_control

		mpc_control = MPCController(dyn_model, horizon, steps_per_episode, 0.02, num_control_samples, 'cc',
							print_minimal)

	# saver = tf.compat.v1.train.Saver(max_to_keep=5)  # WAS 0 BEFORE
	# restore_path = save_dir + '/models/finalModel.ckpt'
	# saver.restore(sess, restore_path)

	#
	print('No issue here')
	rospy.init_node("pub_sub")
	PubSub.sub11callback = sub11_callback
	#
	# controller = PubSub("chatter", "steps", "current_x", "current_y", "current_z", "desired_location",
	#  "desired_y", "desired_z", "current_yaw", "crashed", "inbound", 'roll', 'pitch',
	#  'sim_satus', "pause", "node_status",  "velocity_x", "velocity_y", "velocity_z",
	#  						"angvelocity_x", "angvelocity_y", "angvelocity_z" , 1)

	controller = PubSub("drone1_pose", "drone2_pose", "drone3_pose", "drone4_pose",
						"drone1_vel","drone2_vel",  "drone3_vel","drone4_vel",
						"pl_pose", "pl_vel", "desired_wp", "control1", "control2",
						"control3", "control4", "crashed", "inbound", "sim_status",
						"steps", "pause", "node", 1)

	controller.node_status.publish(True)

	while not rospy.is_shutdown():
		pass

	if not PID:
		with open(save_dir + '/training_data/resulting_x.json', 'r') as file:
			loaded_data = json.load(file)
			resulting_multiple_x = [np.array(arr) for arr in loaded_data]
		with open(save_dir + '/training_data/selected_u.json', 'r') as file:
			loaded_data = json.load(file)
			selected_multiple_u = [np.array(arr) for arr in loaded_data]
		#
		# # with open(save_dir + '/training_data/episode_steps_iter_' + str(counter_agg) + '.json', 'r') as file:
		# # 	loaded_data = json.load(file)
		# # 	loaded_data = [np.array(arr) for arr in loaded_data]
		#
		# resulting_multiple_x = np.reshape(resulting_multiple_x, (resulting_multiple_x.shape[0]*resulting_multiple_x.shape[1],resulting_multiple_x.shape[2]))
		# selected_multiple_u = np.reshape(selected_multiple_u, (selected_multiple_u.shape[0]*selected_multiple_u.shape[1],selected_multiple_u.shape[2]))

		rollouts_forTraining = math.ceil(num_trajectories_for_aggregation*0.8)
		rollouts_forTraining = len(resulting_multiple_x)
		num_trajectories_for_aggregation = 3
		# aggregate data and save it
		if True:
			print("saves new data")

			##############################
			### aggregate some rollouts into training set
			##############################

			# x_array = np.array(resulting_multiple_x)[0:(rollouts_forTraining + 1)]
			x_array = [arr for arr in resulting_multiple_x]
			x_array = x_array[0:(rollouts_forTraining + 1)]
			u_array = [arr for arr in selected_multiple_u]
			u_array = u_array[0:(rollouts_forTraining + 1)]
			# u_array = np.array(selected_multiple_u)[0:(rollouts_forTraining + 1)]
			# u_array = np.squeeze(np.array(selected_multiple_u))[0:(rollouts_forTraining + 1)]
			for i in range(rollouts_forTraining):

				x = x_array[i]  # [N+1, NN_inp]
				u = u_array[i]  # [N, actionSize]

				newDataX = np.copy(x[5:-5, :57])
				newDataY = np.copy(u[5:-5, :])
				newDataZ = np.copy(x[6:-4, :57] - x[5:-5, :57])


				if counter_agg == 0 and i == 0:
					dataX_new = np.zeros((0, mean_x.shape[0]))  # (0, observation space size)
					dataY_new = np.zeros((0, mean_y.shape[0]))
					dataZ_new = np.zeros((0, mean_z.shape[0]))
				elif i < 1:
					dataX_new = np.load(save_dir + '/training_data/dataX_new_iter' + str(counter_agg) + '.npy')
					dataY_new = np.load(save_dir + '/training_data/dataY_new_iter' + str(counter_agg) + '.npy')
					dataZ_new = np.load(save_dir + '/training_data/dataZ_new_iter' + str(counter_agg) + '.npy')
				else:
					dataX_new = np.load(save_dir + '/training_data/dataX_new_iter' + str(counter_agg + 1) + '.npy')
					dataY_new = np.load(save_dir + '/training_data/dataY_new_iter' + str(counter_agg + 1) + '.npy')
					dataZ_new = np.load(save_dir + '/training_data/dataZ_new_iter' + str(counter_agg + 1) + '.npy')

				# the actual aggregation
				dataX_new = np.concatenate((dataX_new, newDataX))
				dataY_new = np.concatenate((dataY_new, newDataY))
				dataZ_new = np.concatenate((dataZ_new, newDataZ))

				np.save(save_dir + '/training_data/dataX_new_iter' + str(counter_agg+1) + '.npy', dataX_new)
				np.save(save_dir + '/training_data/dataY_new_iter' + str(counter_agg+1) + '.npy', dataY_new)
				np.save(save_dir + '/training_data/dataZ_new_iter' + str(counter_agg+1) + '.npy', dataZ_new)


if __name__ == '__main__':
	main()
