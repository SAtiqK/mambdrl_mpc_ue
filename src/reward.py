import numpy as np

from numpy import linalg as LA
import tensorflow as tf

def reward(diff, velo, scores, crashed, inbound, action, prev_rew, multiple):
    # proximity_reward = 1/(LA.norm(state[:3])+0.00001)
    # velocity = [vel.x, vel.y, vel.z]

    if multiple == True:
        vel = velo[:, 0:3]
        ang_vel = velo[:,3:6]
        proximity_reward = np.zeros(diff.shape[0])
        proximity_reward = np.transpose(proximity_reward)
        velocity_reward = np.zeros(diff.shape[0])
        velocity_reward = np.transpose(velocity_reward)
        ang_velocity_reward = np.zeros(diff.shape[0])
        ang_velocity_reward = np.transpose(ang_velocity_reward)
        overshooting_penalty = np.zeros(diff.shape[0])
        overshooting_penalty = np.transpose(overshooting_penalty)


        input_reward = np.zeros(diff.shape[0])
        input_reward = np.transpose(input_reward)


        for i in range(diff.shape[0]):
            temp_z  = diff[i][2]/1000
            # temp = [x/10000 for x in diff[i][0:2]]
            # diff[i][2] = temp_z
            # diff[i][0:2] = temp
            # for j in range(3):
            #     if diff[i][j]>0 and vel[i][j]<0:
            #         velocity_reward[i] += 0.2
            #     elif diff[i][j]>0 and vel[i][j]<0:
            #         velocity_reward[i] += 0.2
            #     else:
            #         velocity_reward[i] += 0

            diffx = abs(diff[i][0]/1000)
            diffy = abs(diff[i][1]/1000)
            diffz = abs(diff[i][2]/300)

            # proximity_reward[i] = (LA.norm([x for x in diff[i][0:3]])**2)/10000000 + diffx + diffz + diffy
            proximity_reward[i] = (LA.norm([x for x in diff[i][0:3]])/7000) ** 2
            velocity_reward[i] = (LA.norm([x for x in vel[i][:]])/1500)**2
            # if LA.norm([x for x in diff[i][0:3]]) < 500:
            #     proximity_reward[i] -= -0.02*LA.norm([x for x in diff[i][0:3]]) + 10
            # if LA.norm([x for x in diff[i][0:3]]) < 2000:
            #     if LA.norm([x for x in vel[i][:]]) > 500:
            #         velocity_reward[i] +=0.3
            # if LA.norm([x for x in diff[i][0:3]]) < 1000:
            #     if LA.norm([x for x in vel[i][:]]) > 300:
            #         velocity_reward[i] += 0.5
            # if LA.norm([x for x in diff[i][0:3]]) < 200:
            #     proximity_reward[i] -= -0.2 * LA.norm([x for x in diff[0:3]]) + 100
            #     if LA.norm([x for x in vel[i][:]]) > 100:
            #         velocity_reward[i] += 25
            # if LA.norm([x for x in diff[i][0:3]]) < 100:
            #     proximity_reward[i] += -100
            #     if LA.norm([x for x in vel[i][:]]) > 20:
            #         velocity_reward[i] += 50
            # ang_velocity_reward[i] = LA.norm([x/50 for x in ang_vel[i][:]])**2
            ang_velocity_reward[i] = abs(vel[i][2]/200)

            if LA.norm([x for x in diff[i][0:3]]) < 100:
                overshooting_penalty[i] = (LA.norm([x for x in vel[i][:]]) / 10) ** 2
            elif LA.norm([x for x in diff[i][0:3]]) < 200:
                # vel_slow = -10 * (LA.norm([x/100 for x in vel[i][:]])**2)
                # pos_rew = -100 * (LA.norm([x for x in diff[i][0:3]])**2)
                overshooting_penalty[i] = (LA.norm([x for x in vel[i][:]])/100)**2
            else:
                overshooting_penalty[i] = 0
            # if np.array(action).any() == 1000:
            #     input_reward[i]= 0
            # else:
            input_reward[i] = LA.norm([x for x in action[i][:]])**2
    else:
        vel = velo[0:3]
        ang_vel = velo[3:6]

        diffx = abs(diff[0] / 1000)
        diffy = abs(diff[1] / 1000)
        diffz = abs(diff[2] / 300)
        # proximity_reward = (LA.norm([x for x in diff[0:3]])**2)/10000000 + diffx + diffz + diffy
        proximity_reward = (LA.norm([x for x in diff[0:3]])/7000) ** 2
        velocity_reward = (LA.norm([x for x in vel])/1500)**2
        # ang_velocity_reward = LA.norm([x / 50 for x in ang_vel[:]]) ** 2
        ang_velocity_reward = abs(vel[2] / 200)
        # if LA.norm([x for x in diff[0:3]]) < 500:
        #     proximity_reward -= -0.02 * LA.norm([x for x in diff[0:3]]) + 10
        # if LA.norm([x for x in diff[0:3]]) < 2000:
        #     if LA.norm([x for x in vel[:]]) > 500:
        #         velocity_reward += 0.3
        # if LA.norm([x for x in diff[0:3]]) < 1000:
        #     if LA.norm([x for x in vel[:]]) > 300:
        #         velocity_reward += 0.5
        # if LA.norm([x for x in diff]) < 5000:
        #     vel_slow = -10 * (LA.norm([x / 100 for x in vel]) ** 2)
        #     pos_rew = -100 * (LA.norm([x for x in diff]) ** 2)
        # if LA.norm([x for x in diff[0:3]]) < 200:
        #     proximity_reward -= -0.2 * LA.norm([x for x in diff[0:3]]) + 100
        #     if LA.norm([x for x in vel[:]]) > 100:
        #         velocity_reward += 25
        # if LA.norm([x for x in diff[0:3]]) < 100:
        #     proximity_reward += -100
        #     if LA.norm([x for x in vel[:]]) > 20:
        #         velocity_reward += 50
        #     overshooting_penalty = vel_slow + pos_rew
        # if action[0] == 1000:
        #     input_reward = 0
        # else:
        if LA.norm([x for x in diff[0:3]]) < 100:
            overshooting_penalty = (LA.norm([x for x in vel[:]]) / 10) ** 2
        elif LA.norm([x for x in diff[0:3]]) < 200:
            # vel_slow = -10 * (LA.norm([x/100 for x in vel[i][:]])**2)
            # pos_rew = -100 * (LA.norm([x for x in diff[i][0:3]])**2)
            overshooting_penalty= (LA.norm([x for x in vel[:]]) / 100) ** 2
        else:
            overshooting_penalty = 0
        input_reward = LA.norm([x for x in action[:]]) ** 2
    ## add a change in reward penalty
    # rew_change = (proximity_reward - prev_rew)
    if crashed == True:
        crashed_punishment = -10000
    else:
        crashed_punishment = 0
    if inbound == True:
        outbound_punishment = 0
    else:
        outbound_punishment = -5000
    # reward = -proximity_reward - 0.1*velocity_reward - 0.001*input_reward
    reward = -proximity_reward - 0.1*velocity_reward
    scores += reward
    return reward, scores

def reward_tf(diff, velo, scores, crashed, inbound, action, prev_rew, multiple):
    # proximity_reward = 1/(LA.norm(state[:3])+0.00001)
    # velocity = [vel.x, vel.y, vel.z]
    proximity_reward = tf.square(tf.norm(diff, axis = 2))/1e7
    velocity_reward = tf.square(tf.norm(velo, axis = 2))/3000

    reward =  -proximity_reward - 0.1 * velocity_reward
    scores = tf.reduce_sum(reward, axis = 0)

    return reward, scores

def reward_payload(drone_pos, diff, velo, pl_velo, pl_orient,  scores, crashed, inbound, action, prev_rew, multiple):
    # proximity_reward = 1/(LA.norm(state[:3])+0.00001)
    # velocity = [vel.x, vel.y, vel.z]

    if multiple == True:
        vel = velo[:, 0:3]
        pl_vel = pl_velo[:,0:3]
        proximity_reward = np.zeros(diff.shape[0])
        proximity_reward = np.transpose(proximity_reward)
        velocity_reward = np.zeros(diff.shape[0])
        velocity_reward = np.transpose(velocity_reward)
        pl_velocity_reward = np.zeros(diff.shape[0])
        pl_velocity_reward = np.transpose(pl_velocity_reward)
        ang_pl_reward = np.zeros(diff.shape[0])
        ang_pl_reward = np.transpose(ang_pl_reward)
        overshooting_penalty = np.zeros(diff.shape[0])
        overshooting_penalty = np.transpose(overshooting_penalty)


        input_reward = np.zeros(diff.shape[0])
        input_reward = np.transpose(input_reward)


        for i in range(diff.shape[0]):
            # pitch = np.radians(pl_orient[i][0])
            # roll = np.radians(pl_orient[i][1])
            # c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
            # c_roll, s_roll = np.cos(roll), np.sin(roll)
            # Rx = np.array(((1, 0, 0), (0, c_roll, -s_roll), (0, s_roll, c_roll)))
            # Ry = np.array(((c_pitch, 0, s_pitch), (0, 1, 0), (-s_pitch, 0, c_pitch)))
            # R = np.matmul(Rx, Ry)
            # pl_z = np.matmul(R, [0, 0, 1])
            # cross_pd = np.dot([0, 0, 1], pl_z)

            # proximity_reward[i] = (LA.norm([x for x in diff[i][0:3]])**2)/10000000 + diffx + diffz + diffy
            # proximity_reward[i] = (LA.norm([x for x in diff[i][0:2]])/7000) ** 2 + 2*((LA.norm( diff[i][2])/2500) ** 2)
            proximity_reward[i] = (LA.norm([x for x in diff[i][0:3]])/7000) ** 2
            velocity_reward[i] = (LA.norm([x for x in vel[i][:]])/1500)**2
            # pl_velocity_reward[i] = (LA.norm([x for x in pl_vel[i][:]])/100)**2
            pl_velocity_reward[i] = (LA.norm([x for x in action[i][:]]))
            # pl_posi = np.arccos(120 / LA.norm([pl_orient[i][0:1], drone_pos[i][0:1]]))
            pl_error = [drone_pos[i][0], drone_pos[i][1], drone_pos[i][2] -100] - pl_orient[i]
            # pl_error = [drone_pos[i][0], drone_pos[i][1]] - pl_orient[i][0:2]
            ang_pl_reward[i] = (LA.norm([x for x in pl_error])/100) ** 2
            # ang_pl_reward[i] = (LA.norm([x for x in pl_orient[i][:]])/90)**2
            # ang_velocity_reward[i] = abs(vel[i][2]/200)
            # ang_pl_reward[i] = 1/abs(pl_posi)

    else:
        vel = velo[0:3]
        pl_vel = pl_velo[0:3]

        # pitch = np.radians(pl_orient[0])
        # roll = np.radians(pl_orient[1])
        # c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        # c_roll, s_roll = np.cos(roll), np.sin(roll)
        # Rx = np.array(((1, 0, 0), (0, c_roll, -s_roll), (0, s_roll, c_roll)))
        # Ry = np.array(((c_pitch, 0, s_pitch), (0, 1, 0), (-s_pitch, 0, c_pitch)))
        # R = np.matmul(Rx, Ry)
        # pl_z = np.matmul(R, [0, 0, 1])
        # cross_pd = np.dot([0, 0, 1], pl_z)

        # proximity_reward = (LA.norm([x for x in diff[0:3]])**2)/10000000 + diffx + diffz + diffy
        # proximity_reward = (LA.norm([x for x in diff[0:2]])/7000) ** 2 + 2*((LA.norm( diff[2])/2500) ** 2)
        proximity_reward = (LA.norm([x for x in diff[0:3]]) / 7000) ** 2
        velocity_reward = (LA.norm([x for x in vel])/1500)**2
        # pl_velocity_reward = (LA.norm([x for x in pl_vel[:]]) / 100) ** 2
        pl_velocity_reward = (LA.norm([x for x in action[:]]))
        # ang_pl_reward = (LA.norm([x for x in pl_orient]) / 90) ** 2
        pl_error = [drone_pos[0], drone_pos[1], drone_pos[2] - 100] - pl_orient
        # pl_error = [drone_pos[0], drone_pos[1]] - pl_orient[0:2]
        ang_pl_reward = (LA.norm([x for x in pl_error])/100) ** 2
        # print("Cross product:" + str(cross_pd))
        # pl_posi = np.arccos(120/LA.norm([pl_orient[0:1], drone_pos[0:1]]))
        # ang_pl_reward[i] = (LA.norm([x for x in pl_orient[i][:]])/90)**2
        # ang_velocity_reward[i] = abs(vel[i][2]/200)
        # ang_pl_reward = 1/abs(pl_posi)
        # ang_velocity_reward = LA.norm([x / 50 for x in ang_vel[:]]) ** 2
        ang_velocity_reward = abs(vel[2] / 200)
# add the taught factor in the reward function + orientation in each axis

    reward = -proximity_reward - 0.1*velocity_reward
    # reward = -proximity_reward - 0.1*velocity_reward - 0.5*pl_velocity_reward
    # reward = - proximity_reward - 0.1*velocity_reward - 0.5*ang_pl_reward
    # g_pl_reward - 1.1* pl_velocity_reward
    # reward = - ang_pl_reward - 0.05*pl_velocity_reward
    scores += reward
    return reward, scores, [proximity_reward, velocity_reward, ang_pl_reward]

def reward_multiagent(observation, diff,  scores, action, multiple):
    # proximity_reward = 1/(LA.norm(state[:3])+0.00001)
    # velocity = [vel.x, vel.y, vel.z]

    if multiple == True:
        vel = observation[:, 6:9]
        drone_pos = observation[:, 0:3]
        pl_orient = observation[:, 51:54]
        pl_vel = observation[:,54:57]
        proximity_reward = np.zeros(diff.shape[0])
        proximity_reward = np.transpose(proximity_reward)
        velocity_reward = np.zeros(diff.shape[0])
        velocity_reward = np.transpose(velocity_reward)
        pl_velocity_reward = np.zeros(diff.shape[0])
        pl_velocity_reward = np.transpose(pl_velocity_reward)
        ang_pl_reward = np.zeros(diff.shape[0])
        ang_pl_reward = np.transpose(ang_pl_reward)
        overshooting_penalty = np.zeros(diff.shape[0])

        input_reward = np.zeros(diff.shape[0])
        input_reward = np.transpose(input_reward)


        for i in range(diff.shape[0]):
            proximity_reward[i] = (LA.norm([x for x in diff[i][0:3]])/7000) ** 2
            velocity_reward[i] = (LA.norm([x for x in vel[i][:]])/1500)**2
            # pl_velocity_reward[i] = (LA.norm([x for x in pl_vel[i][:]])/100)**2
            pl_velocity_reward[i] = (LA.norm([x for x in action[i][:]]))
            # pl_posi = np.arccos(120 / LA.norm([pl_orient[i][0:1], drone_pos[i][0:1]]))
            pl_error = [drone_pos[i][0], drone_pos[i][1], drone_pos[i][2] -100] - pl_orient[i]
            # pl_error = [drone_pos[i][0], drone_pos[i][1]] - pl_orient[i][0:2]
            ang_pl_reward[i] = (LA.norm([x for x in pl_error])/100) ** 2

    else:
        vel = observation[6:9]
        pl_vel = observation[48:51]
        drone_pos = observation[0:3]
        pl_orient = observation[51:54]

        # proximity_reward = (LA.norm([x for x in diff[0:3]])**2)/10000000 + diffx + diffz + diffy
        # proximity_reward = (LA.norm([x for x in diff[0:2]])/7000) ** 2 + 2*((LA.norm( diff[2])/2500) ** 2)
        proximity_reward = (LA.norm([x for x in diff[0:3]]) / 7000) ** 2
        velocity_reward = (LA.norm([x for x in vel])/1500)**2
        # pl_velocity_reward = (LA.norm([x for x in pl_vel[:]]) / 100) ** 2
        # pl_velocity_reward = (LA.norm([x for x in action[:]]))
        # ang_pl_reward = (LA.norm([x for x in pl_orient]) / 90) ** 2
        # pl_error = [drone_pos[0], drone_pos[1], drone_pos[2] - 100] - pl_orient
        # pl_error = [drone_pos[0], drone_pos[1]] - pl_orient[0:2]
        # ang_pl_reward = (LA.norm([x for x in pl_error])/100) ** 2


    # reward = -proximity_reward - 0.1*velocity_reward - ang_pl_reward
    reward = -proximity_reward
    # reward = -proximity_reward - 0.1*velocity_reward - 0.5*pl_velocity_reward
    # reward = - proximity_reward - 0.1*velocity_reward - 0.5*ang_pl_reward
    # g_pl_reward - 1.1* pl_velocity_reward
    # reward = - ang_pl_reward - 0.05*pl_velocity_reward
    scores += reward
    return reward, scores
            # [proximity_reward, velocity_reward, ang_pl_reward])