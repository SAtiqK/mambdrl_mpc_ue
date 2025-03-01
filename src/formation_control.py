#!/usr/bin/env python3
import numpy as np

def position_based_fc(observation, PID):
    #add code for each drone frame coordinates
    offset = [[0, -140, 0], [-140, 0, 0], [-140, -140, 0]]
    follower_poses = [observation[12:18], observation[24:30], observation[36:42]]
    leader_position = observation[0:3]
    inputs = []

    for i in range(3):
        x_rotated_curr = follower_poses[i][0] * np.cos(np.radians(-follower_poses[i][5])) - follower_poses[i][1] * np.sin(
            np.radians(-follower_poses[i][5]))
        y_rotated_curr = follower_poses[i][0] * np.sin(np.radians(-follower_poses[i][5])) +  follower_poses[i][1] * np.cos(
            np.radians(-follower_poses[i][5]))

        x_rotated_des = leader_position[0] * np.cos(np.radians(-follower_poses[i][5])) - leader_position[1] * np.sin(
            np.radians(-follower_poses[i][5]))
        y_rotated_des = leader_position[0] * np.sin(np.radians(-follower_poses[i][5])) + leader_position[1] * np.cos(
            np.radians(-follower_poses[i][5]))
        desired_position = [x_rotated_des + offset[i][0], y_rotated_des + offset[i][1]]

        control_y = PID.compute(desired_position[0], x_rotated_curr, 1)
        control_x = PID.compute(desired_position[1], y_rotated_curr, 1)
        control_z = 0
        control_thrust = PID.compute(leader_position[2], follower_poses[i][2], 1)

        inputs.append([control_x, control_y,control_z,control_thrust])

    return inputs