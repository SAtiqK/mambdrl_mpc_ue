import rospy
import json
import copy
from geometry_msgs.msg import Quaternion, Vector3
from payload_datagather import UEmsgs_to_observations
from payload_datagather import check_terminal_state
import numpy as np
from data_save import data_save
from data_manipulation import *
from numpy import linalg as LA
import tensorflow as tf


pController_freq = 2
episode_counter = 0


def UEmsgs_to_observations_first(current_state, desired_state, current_vel):
	current_position = [current_state.x, current_state.y, current_state.z, current_state.yaw, current_state.pitch, current_state.roll]
	current_velocity = [current_vel.x, current_vel.y, current_vel.z, current_vel.yaw, current_vel.pitch, current_vel.roll]
	desired_state = [desired_state.x, desired_state.y, desired_state.z]
	diff = [desired_state[i] - current_position[i] for i in range(min(len(desired_state), len(current_position)))]
	x_rotated_curr = current_state.x * np.cos(np.radians(-current_state.yaw)) - current_state.y * np.sin(np.radians(-current_state.yaw))
	y_rotated_curr = current_state.x * np.sin(np.radians(-current_state.yaw)) + current_state.y * np.cos(np.radians(-current_state.yaw))

	x_rotated_des = desired_state[0] * np.cos(np.radians(-current_state.yaw)) - desired_state[1] * np.sin(
		np.radians(-current_state.yaw))
	y_rotated_des = desired_state[0] * np.sin(np.radians(-current_state.yaw)) + desired_state[1] * np.cos(
		np.radians(-current_state.yaw))
	diff_df = [(x_rotated_des-x_rotated_curr), (y_rotated_des-y_rotated_curr), desired_state[2]-current_state.z]
	# observation = np.copy(current_position)
	observation = np.concatenate((current_position, current_velocity))
	# observation.append(current_velocity)
	return current_position, desired_state, current_velocity, observation, diff, diff_df

def UEmsgs_to_observations_tf(current_state, desired_state, current_vel):
	current_position = tf.stack([current_state[..., 0], current_state[..., 1], current_state[..., 2], current_state[..., 3],
		 current_state[..., 4], current_state[..., 5]], axis=-1)
	current_velocity = tf.stack([current_vel[..., 0], current_vel[..., 1], current_vel[..., 2], current_vel[..., 3], current_vel[..., 4],
		 current_vel[..., 5]], axis=-1)
	desired_state = tf.stack([desired_state[..., 0], desired_state[..., 1], desired_state[..., 2], desired_state[..., 3]], axis = -1)

	diff = desired_state - current_position[0:4]

	observation = tf.concat([current_position, current_velocity], axis=-1)
	return current_position, desired_state, current_velocity, observation, diff


def check_terminal_state_first(diff, velocity, crashed, inbound, steps):
	# check if the drone has reached the final state - check from the desired state and current state thing
	# if abs(diff[0]) < 200 and abs(diff[1]) < 200 and abs(diff[2]) < 200 and abs(diff[3]) < 90:
	# print(LA.norm([diff[0], diff[1], diff[2]]))
	if LA.norm([diff[0], diff[1], diff[2]]) < 100 and LA.norm([velocity[0:3]])<10:
		terminal = 1
		reached = True
	# check if the drone is outside of the bounds - check from Unreal
	elif crashed == 1 or inbound == 0 or steps == 1:
		terminal = 1
		reached = False
	else:
		terminal = 0
		reached = False
	# check if the drone has crashed - check from Unreal
	return terminal, reached

def train(self, current_s, desired_s, current_vel, pl_pos, pl_vel, crashed, inbound, info, traindata, expl_policy, MultiAgent):
	#needs to check for terminal state - DONE
	prvs_terminal = info.prev_terminal
	a = Quaternion()
	if MultiAgent:
		a.x = 0
		a.y = 0
		a.z = 0
		a.w = 0
		action = [a, a, a, a]
	else:
		a.x = 0
		a.y = 0
		a.z = 0
		a.w = 0
	observation = np.array([])
	for i in range(len(current_s)):
		if i == 3:
			current_state, desired_state, current_velocity, current_position_pl, current_velocity_pl, _,diff, _ = UEmsgs_to_observations(
				current_s[i], desired_s, current_vel[i], pl_pos, pl_vel)
			observation = np.concatenate((observation, current_state))
			observation = np.concatenate((observation, current_velocity))
			observation = np.concatenate((observation, current_position_pl))
			observation = np.concatenate((observation, current_velocity_pl[0:3]))
		else:
			current_state, _, current_velocity, _, _, _, _,_ = UEmsgs_to_observations(
				current_s[i], desired_s, current_vel[i], pl_pos, pl_vel)
			observation = np.concatenate((observation, current_state))
			observation = np.concatenate((observation, current_velocity))

	terminal, reached = check_terminal_state(diff, current_position_pl, current_velocity_pl, crashed, inbound,
											 info.steps)
	# current_state, desired_state, current_velocity, observation, diff, diff_dr = UEmsgs_to_observations(current_s, desired_s, current_vel)
	# terminal, reached = check_terminal_state(diff, current_velocity, crashed, inbound, info.steps)
	# expl_policy = random_policy(pController_freq, current_xyz, desired_xyz, episode_counter)

	if terminal == 1:
		if prvs_terminal == False:
			print('Steps:' + str(info.steps))
			print('InBound:' + str(inbound))
			print('Crashed:' + str(crashed))
			print('Reached: ' + str(reached))
			if info.episode_counter <= info.episode_number:
				info.episode_counter = info.episode_counter + 1
				print("Episode " + str(info.episode_counter))
			else:
				info.episode_val_counter = info.episode_val_counter + 1
				print("Episode " + str(info.episode_val_counter))
			print("DONE TAKING ", info.steps_rollout_counter, " STEPS.")
			traindata.waypoint.append(np.array(desired_state))
			traindata.epi_observations.append(np.array(traindata.observations))
			traindata.epi_actions.append(np.array(traindata.actions))
			traindata.observations = []
			traindata.actions = []
			# traindata.observations.pop()
			# traindata.actions.pop()
		else:
			pass
		info.steps_rollout_counter = 0
	else:
		if info.episode_counter <= info.episode_number:
			offset = [[70,70], [70, -70], [-70, 70], [-70,-70]]
			desired_position = Vector3()
			inputs = np.array([])
			action = []
			for i in range(len(current_s)):
				desired_position.x = np.copy(desired_s.x) + offset[i][0]
				desired_position.y = np.copy(desired_s.y) + offset[i][1]
				desired_position.z = np.copy(desired_s.z) + 150
				control_inp, control = expl_policy.exploration_policy(current_s[i], desired_position)
				inputs = np.concatenate((inputs, control_inp))
				action.append(copy.deepcopy(control))
			traindata.observations.append(observation)
			traindata.actions.append(np.array(inputs))
			# if info.steps_rollout_counter == 0: #saves output and skips the first step in a rollout
			# 	pass
			# else:
			# 	l = len(traindata.observations)
			# 	output = np.copy([traindata.observations[l-1][i] - traindata.observations[l-2][i] for i in range(len(observation))])
			# 	traindata.output.append(output)

		else:
			if (info.episode_val_counter == 0) and (info.steps_rollout_counter == 0):
				if MultiAgent:

					a.x = 0
					a.y = 0
					a.z = 0
					a.w = 0
					action = [a, a, a, a]
				else:
					a.x = 0
					a.y = 0
					a.z = 0
					a.w = 0
					action = a

				dataX, dataY = generate_training_data_inputs(traindata.epi_observations, traindata.epi_actions)
				dataZ = generate_training_data_outputs(traindata.epi_observations)
				np.save(traindata.save_dir + '/training_data/dataX.npy', dataX)
				np.save(traindata.save_dir + '/training_data/dataY.npy', dataY)
				np.save(traindata.save_dir + '/training_data/dataZ.npy', dataZ)
				np.save(traindata.save_dir + '/training_data/training_waypoints.npy', traindata.waypoint)
				traindata.observations = []
				traindata.actions = []
				traindata.output = []
				traindata.epi_observations = []
				traindata.epi_actions = []
			if info.episode_val_counter <= info.episode_val:
				inputs, action = expl_policy.exploration_policy()
				traindata.observations.append(observation.tolist())
				traindata.actions.append(inputs)
				# if info.steps_rollout_counter == 0:  # saves output and skips the first step in a rollout
				# 	pass
				# else:
				# 	l = len(traindata.observations)
				# 	output = [traindata.observations[l - 1][i] - traindata.observations[l - 2][i] for i in range(len(observation))]
				# 	traindata.output.append(output)
			else:
				print('here')
				if MultiAgent:

					a.x = 0
					a.y = 0
					a.z = 0
					a.w = 0
					action = [a, a, a, a]
				else:
					a.x = 0
					a.y = 0
					a.z = 0
					a.w = 0
					action = a

				# dataX_val, dataY_val = generate_training_data_inputs(traindata.observations, traindata.actions)
				# dataZ_val = generate_training_data_outputs(traindata.observations)
				with open(traindata.save_dir + '/training_data/states_val.json', 'w') as file:
					serialized_data = [arr.tolist() for arr in traindata.epi_observations]
					json.dump(serialized_data, file)
				with open(traindata.save_dir + '/training_data/controls_val.json', 'w') as file:
					serialized_data = [arr.tolist() for arr in traindata.epi_actions]
					json.dump(serialized_data, file)
				# np.save(traindata.save_dir + '/training_data/dataZ_val.npy', dataZ_val)
				rospy.signal_shutdown("Done")

		#print all the results and shut down the ROS node
		info.steps_rollout_counter = info.steps_rollout_counter + 1
		# print(info.steps_rollout_counter)
	if info.steps_rollout_counter < info.steps_rollout and info.steps_rollout_counter > 0:
		info.steps = bool(0)
		# print('cuz of steps')
	elif prvs_terminal == False:
		info.steps = bool(1)
		self.steps.publish(info.steps)
	elif info.simu_status == 1:
		info.steps = bool(0)
		# print('cause of ss')
	else:
		info.steps = bool(1)
	info.prev_terminal = bool(terminal)

	return action

# def get_mpc_controller():
