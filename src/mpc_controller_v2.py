#!/usr/bin/env python3
import numpy.random as npr
from geometry_msgs.msg import Quaternion

from PID_controller import PIDController
from payload_datagather import UEmsgs_to_observations
from data_save import data_save_mpc
import rospy
import json
from formation_control import position_based_fc
import tensorflow as tf
import os
from reward import reward
from reward import *
from payload_datagather import check_terminal_state
from episode_info import episodeInfoMPC
import torch
inputs = Quaternion()
traindata = data_save_mpc(1)


class MPCController:

    def __init__(self, dyn_model, horizon, steps_per_episode, dt_steps, num_control_samples, actions_ag, print_minimal):

        #init vars
        self.N = num_control_samples
        self.horizon = horizon
        self.dyn_model = dyn_model
        self.steps_per_episode = steps_per_episode
        self.actions_ag = actions_ag
        self.print_minimal = print_minimal
        self.steps = False
        self.epi_info = episodeInfoMPC()
        self.epsiodes_total = self.epi_info.episode_number
        self.fc_PID = PIDController(0.005, 0, 0)
        self.fc_freq = 4


    def rollout_step(self, pub, current, desired, vel, pl_pos, pl_vel, inbound, crashed, save_dir, counter_agg, restored_model, PETS, Multi_Agent, combined_state):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        gpu_device = 1
        gpu_frac = 0.9
        # >>>>>>> a0465618a9a1bcee1d4b6d6aee9813ef04e6ba44

        # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        inputs = Quaternion()

        prvs_terminal = self.epi_info.prev_terminal
        # print(prvs_terminal)
        episode_counter = self.epi_info.episode_counter
        global wp
        observation = []
        # current_state, desired_state, current_velocity, observation, diff , diff_df= UEmsgs_to_observations(current, desired, vel)
        # terminal, reached = check_terminal_state(diff, current_velocity, pl_vel, crashed, inbound, self.epi_info.steps)
        if Multi_Agent:
            for i in range(len(combined_state)):
                if i == 3:
                    current_state, desired_state, current_velocity, current_position_pl, current_velocity_pl, _, diff, diff_dr = UEmsgs_to_observations(
                        combined_state[i], desired, vel[i], pl_pos, pl_vel)
                    observation.extend(current_state)
                    observation.extend(current_velocity)
                    observation.extend(current_position_pl)
                    observation.extend(current_velocity_pl)
                else:
                    current_state, _, current_velocity, _, _, _, _, _ = UEmsgs_to_observations(
                        combined_state[i], desired, vel[i], pl_pos, pl_vel)
                    observation.extend(current_state)
                    observation.extend(current_velocity)
            combined_observation = observation
            terminal, reached = check_terminal_state(diff, current_position_pl, current_velocity_pl, crashed, inbound,
                                                     self.epi_info.steps)
        else:
            for i in range(len(combined_state)):
                if i == 3:
                    current_state, desired_state, current_velocity, current_position_pl, current_velocity_pl, _, diff, diff_dr = UEmsgs_to_observations(
                        combined_state[i], desired, vel, pl_pos, pl_vel)
                    observation.extend(current_state)
                    observation.extend(current_velocity)
                    observation.extend(current_position_pl)
                    observation.extend(current_velocity_pl)
                else:
                    current_state, _, current_velocity, _, _, _, _, _ = UEmsgs_to_observations(
                        combined_state[i], desired, vel, pl_pos, pl_vel)
                    observation.extend(current_state)
                    observation.extend(current_velocity)
            combined_observation = observation
            current_state, desired_state, current_velocity, current_position_pl, current_velocity_pl, observation, diff, diff_dr = UEmsgs_to_observations(current, desired, vel, pl_pos, pl_vel)
            terminal, reached = check_terminal_state(diff, current_velocity, current_velocity_pl, crashed, inbound,
                                                 self.epi_info.steps)

        # accumulate reward over each timestep
        if terminal == 1:
            if prvs_terminal == False:
                print('Steps:' + str(self.epi_info.steps))
                print('InBound:' + str(inbound))
                print('Crashed:' + str(crashed))
                self.epi_info.episode_counter = episode_counter + 1
                print("Episode " + str(self.epi_info.episode_counter))
                # step_reward, _ = reward(observation, current_velocity,np.array([0, 0, 0]), crashed, inbound, 1000, traindata.prev_rew, False )
                # traindata.episode_reward += step_reward
                if (not (self.print_minimal)):
                    print("DONE TAKING ", self.epi_info.steps_rollout_counter, " STEPS.")
                    print("Reward: ", traindata.episode_reward)

                traindata.waypoint.append(np.array(wp))
                np.save(save_dir + '/training_data/mpc_waypoints_' + str(counter_agg) + '.npy', traindata.waypoint)

                traindata.resulting_multiple_x.append(traindata.list_episode_observations)
                traindata.selected_multiple_u.append(traindata.list_episode_actions)
                traindata.total_episode_reward.append(traindata.episode_reward)
                traindata.total_episode_steps.append(self.epi_info.steps_rollout_counter)
                x = traindata.resulting_multiple_x
                y = traindata.selected_multiple_u
                with open(save_dir + '/training_data/resulting_x.json', 'w') as file:
                    serialized_data = [arr for arr in traindata.resulting_multiple_x]
                    json.dump(serialized_data, file)
                with open(save_dir + '/training_data/selected_u.json', 'w') as file:
                    serialized_data = [arr for arr in traindata.selected_multiple_u]
                    json.dump(serialized_data, file)
                # with open(save_dir + '/training_data/episode_rewards_iter_' + str(counter_agg) + '.json', 'w') as file:
                #     serialized_data = [arr for arr in traindata.total_episode_reward]
                #     json.dump(serialized_data, file)
                # with open(save_dir + '/training_data/episode_steps_iter_' + str(counter_agg) + '.json', 'w') as file:
                #     serialized_data = [arr for arr in traindata.total_episode_steps]
                #     json.dump(serialized_data, file)
                # # np.save(save_dir + '/training_data/resulting_x.npy', traindata.resulting_multiple_x)
                # np.save(save_dir + '/training_data/selected_u.npy', traindata.selected_multiple_u)
                np.save(save_dir + '/training_data/rew_comps_' + str(counter_agg) + str(self.epi_info.episode_counter) + '.npy', traindata.reward_comps)
                np.save(save_dir + '/saved_trajfollow/episode_rewards_iter_' + str(counter_agg) + '.npy', traindata.total_episode_reward)
                np.save(save_dir + '/training_data/episode_steps_iter_' + str(counter_agg) + '.npy', traindata.total_episode_steps)

            else:
                pass
            self.epi_info.steps_rollout_counter = 0
            if self.epi_info.episode_counter >= self.epi_info.episode_number:
                rospy.signal_shutdown('Done')
        else:
            if self.epi_info.episode_counter < self.epi_info.episode_number:
                if self.epi_info.steps_rollout_counter == 0:
                    traindata.list_episode_actions = []
                    traindata.list_episode_observations = []
                    traindata.list_episode_rewards = []
                    traindata.episode_steps = []
                    traindata.reward_comps = []
                    traindata.episode_reward = 0
                    traindata.prev_rew = 0
                if self.epi_info.steps_rollout_counter == 2:
                    wp = desired_state[0:3]
                inputs, best_sim_number, best_sequence, moved_to_next, best_action = self.get_action(observation,
                                                                                                     desired, vel,
                                                                                                     pl_pos, pl_pos,
                                                                                                     restored_model,
                                                                                                     PETS, Multi_Agent, combined_observation)
                if Multi_Agent:
                    step_reward, _ = reward_multiagent(observation, diff, 0, best_action, False)
                else:
                    step_reward, _, rew_comp = reward_payload(observation[0:3], diff, current_velocity,
                                                              current_velocity_pl, observation[12:15], 0, crashed,
                                                              inbound, best_action, traindata.prev_rew, False)
                print("Step Reward:" + str(step_reward))
                traindata.list_episode_observations.append(combined_observation)
                traindata.reward_comps.append(0)
                traindata.list_episode_actions.append(best_action.tolist())
                traindata.list_episode_rewards.append(step_reward)
                traindata.episode_reward += step_reward
                traindata.prev_rew = step_reward
            else:
                inputs.x = 0
                inputs.y = 0
                inputs.z = 0
                inputs.w = 0
                print('Done running given episodes')
                terminal = 1
                pub.steps.publish(True)
                rospy.signal_shutdown('Done')
                # print all the results and shut down the ROS node
            self.epi_info.steps_rollout_counter = self.epi_info.steps_rollout_counter + 1
            if (not (self.print_minimal)):
                if (self.epi_info.steps_rollout_counter % 100 == 0):
                    print("done step ", self.epi_info.steps_rollout_counter, ", rew: ", traindata.episode_reward)
            # print(self.epi_info.steps_rollout_counter)
        if self.epi_info.steps_rollout_counter < self.epi_info.steps_rollout and self.epi_info.steps_rollout_counter > 0:
            self.epi_info.steps = bool(0)
        # print('cuz of steps')
        elif prvs_terminal == False:
            self.epi_info.steps = bool(1)
            pub.steps.publish(self.epi_info.steps)
            print('Steps Over')
        elif self.epi_info.simu_status == 1:
            self.epi_info.steps = bool(0)
            print('cause of ss')
            # pub.steps.publish(self.epi_info.steps)
        else:
            self.epi_info.steps = bool(1)
        # if self.epi_info.episode_counter == 2 and self.epi_info.steps_rollout_counter == 2:
        #     self.epi_info.steps = bool(1)
        #     pub.steps.publish(self.epi_info.steps)
        self.epi_info.prev_terminal = bool(terminal)
        pub.node_status.publish(True)
        return inputs

    def get_action(self, state, desired, vel, pl_pos, pl_vel, restored_model,PETS, Multi_Agent, comb_ob):
        tf.config.run_functions_eagerly(True)
        curr_nn_state = state[0:57]
        comb_ob = comb_ob[0:57]
        ub = 1
        lb = -1
        # all_samples = npr.uniform([-1, -1, -1], [1, 1, 1], (self.N, self.horizon, 3))
        if Multi_Agent:
            all_samples = npr.uniform([lb, lb, 0, lb, lb, lb, 0, lb,lb, lb, 0, lb,lb, lb, 0, lb],
                                      [ub, ub, 0, ub, ub, ub, 0, ub,ub, ub, 0, ub,ub, ub, 0, ub], (self.N, self.horizon, 16))
        else:
            all_samples = npr.uniform([-0.5, -0.5, 0, -1], [0.5, 0.5, 0, 1], (self.N, self.horizon, 4))

        # forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
        many_in_parallel = True
        if PETS:
            resulting_states = self.dyn_model.predict_next_obs(torch.tensor(curr_nn_state), np.copy(all_samples), many_in_parallel)
        else:
            resulting_states = self.dyn_model.do_forward_sim([curr_nn_state, 0], np.copy(all_samples), many_in_parallel, restored_model)
            resulting_states = np.array(resulting_states)  # this is [horizon+1, N, statesize]

        if Multi_Agent:
            resulting_states = np.array(resulting_states[1:])
            desired_array = np.array([desired.x, desired.y, desired.z])
            diff = resulting_states[:,:,48:51] - desired_array[np.newaxis, np.newaxis, :]
        else:
            resulting_states = np.array(resulting_states[1:])
            desired_array = np.array([desired.x, desired.y, desired.z])
            diff = resulting_states[:, :, 0:3] - desired_array[np.newaxis, np.newaxis, :]



        # init vars to evaluate the trajectories
        scores = np.zeros((self.N,))
        # curr_seg = np.tile(curr_line_segment,(self.N,))
        # curr_seg = curr_seg.astype(int)
        moved_to_next = np.zeros((self.N,))
        prev_pt = resulting_states[0]
        prev_r = np.zeros((self.N,))

        # accumulate reward over each timestep
        if self.epi_info.steps_rollout_counter%self.fc_freq == 0:
            inputs = Quaternion()
            for pt_number in range(resulting_states.shape[0]-1):
                # array of "the point"... for each sim
                pt = diff[pt_number]  # N x state
                v = resulting_states[pt_number,:,6:9]
                v_pl = resulting_states[pt_number,:,18:21]
                orient_pl = resulting_states[pt_number,:,15:18]
                drone_pos = resulting_states[pt_number, :, 0:3]

                # if pt_number < 10:
                action = all_samples[:, pt_number, :]

                # else:
                #     action = np.zeros((pt.shape[0],4))
                    # action = np.transpose(action)
                if Multi_Agent:
                    rew, scores = reward_multiagent(resulting_states[pt_number], pt, scores, action, True)
                    # rew, scores = reward(pt, v, scores, False, False, action, prev_r, True)
                else:
                    rew, scores, _ = reward_payload(drone_pos, pt, v, v_pl, orient_pl, scores, False, False, action, prev_r, True)

                prev_r = rew

                if pt_number == 0:
                    exp_rew = np.copy(scores)
                # update vars
                # prev_forward = np.copy(curr_forward)
                prev_pt = np.copy(pt)

            # pick best action sequence
            best_score = np.max(scores)
            best_sim_number = np.argmax(scores)
            best_sequence = all_samples[best_sim_number]
            best_action = np.copy(best_sequence[0])

            inputs.x = best_action[0]
            inputs.y = best_action[1]
            inputs.z = best_action[2]
            # inputs.z = 0
            inputs.w = best_action[3]
            min_action = -1
            max_action = 1
            best_action = np.clip(best_action, min_action, max_action)
            action = []
            for i in range(4):
                if i == 0:
                    inputs.x = max(min(inputs.x, max_action), min_action)
                    inputs.y = max(min(inputs.y, max_action), min_action)
                    inputs.z = max(min(inputs.z, max_action), min_action)
                    inputs.w = max(min(inputs.w, max_action), min_action)
                else:
                    inputs = Quaternion()
                    inputs.x = 0
                    inputs.y = 0
                    inputs.z = 0
                    inputs.w = 0
                action.append(inputs)

            return action, best_sim_number, best_sequence, moved_to_next, best_action
        else:
            control_inputs = position_based_fc(comb_ob, self.fc_PID)
            action = []
            best_action = []
            for i in range(4):
                inputs = Quaternion()
                if i == 0:
                    inputs.x = 0
                    inputs.y = 0
                    inputs.z = 0
                    inputs.w = 0
                else:
                    inputs.x = control_inputs[i-1][0]
                    inputs.y = control_inputs[i-1][1]
                    inputs.z = control_inputs[i-1][2]
                    inputs.w = control_inputs[i-1][3]
                action.append(inputs)
                best_action.append(inputs.x)
                best_action.append(inputs.y)
                best_action.append(inputs.z)
                best_action.append(inputs.w)
            return action, 0, 0, 0, np.array(best_action)

        # print("Expected reward:" + str(exp_rew[best_sim_number]))


    def to_check(self):
        print('inside MPC')
