#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json

from docutils.nodes import label
# import tensorflow as tf
from numpy import linalg as LA

def main():
    save_dir = 'run_5006'

    # counter_agg = 0
    # with open(save_dir + '/training_data/resulting_x.json', 'r') as file:
    #     loaded_data = json.load(file)
    #     loaded_data = [np.array(arr) for arr in loaded_data]
    # with open(save_dir + '/training_data/selected_u.json', 'r') as file:
    #     loaded_data = json.load(file)
    #     loaded_data = [np.array(arr) for arr in loaded_data]
    # with open(save_dir + '/training_data/episode_rewards_iter_' + str(counter_agg) + '.json', 'r') as file:
    #     loaded_data = json.load(file)
    #     loaded_data = [np.array(arr) for arr in loaded_data]
    # with open(save_dir + '/training_data/episode_steps_iter_' + str(counter_agg) + '.json', 'r') as file:
    #     loaded_data = json.load(file)
    #     loaded_data = [np.array(arr) for arr in loaded_data]



    # dataX_new = np.load(save_dir + '/training_data/dataX_new_iter4.npy') # (0, observation space size)
    # dataY_new = np.load(save_dir + '/training_data/dataY_pp.npy') # (0, observation space size)
    # dataZ_new = np.load(save_dir + '/training_data/dataZ.npy') # (0, observation space size)

    x = 1
    # dataY_new = np.load(save_dir + '/training_data/dataY_new_iter' + str(counter_agg) + '.npy')
    # dataZ_new = np.load(save_dir + '/training_data/dataZ_new_iter' + str(counter_agg) + '.npy')

    # rews = np.load(save_dir +'/losses/list_old_loss.npy')
    # print(rews)
    with open(save_dir + '/training_data/resulting_x.json', 'r') as json_file:
        # Load the JSON data into a Python data structure
        data = json.load(json_file)
    # dataX_new = np.load(save_dir + '/training_data/mpc_waypoints_0.npy') # (0, observation space size)
    # waypoints= np.load(save_dir + '/training_data/mpc_waypoints_0.npy') # (0, observation space size)
    # rew_comps = np.load(save_dir + '/training_data/mpc_waypoints_19.npy')
    # dataY= np.load(save_dir + '/training_data/dataX.npy') # (0, observation space size)
    # dataY1= np.load(save_dir + '/training_data/dataZ_pp.npy') # (0, observation space size)
    pass
    if True:
        e = 0
        l = len(data[e])
        s = np.zeros(l)
        x = np.zeros(l)
        y = np.zeros(l)
        z = np.zeros(l)
        dist = np.zeros(l)
        vel = np.zeros(l)
        swing = np.zeros(l)


        current = np.array(data[e])

        xc = current[:,0]
        yc = current[:,1]
        zc = current[:,2]
        xv = current[:,6]
        yv = current[:, 7]
        zv = current[:, 8]
        pl = current[:,48:51]
        xc1 = current[:, 12]
        yc1 = current[:, 13]
        zc1 = current[:, 14]
        xc2= current[:, 24]
        yc2 = current[:, 25]
        zc2 = current[:, 26]
        xc3 = current[:, 36]
        yc3 = current[:, 37]
        zc3 = current[:, 38]

        # for j in range(len(data)):
        #     l = len(data[j])

        for i in range(l):
            s[i] = i
            # x[i] = dataX_new[j][0]
            # y[i] = dataX_new[j][1]
            # z[i] = dataX_new[j][2]
            x[i] = 1500
            y[i] = 10050
            z[i] = 1390
            dist[i] = LA.norm([(x[i] - xc[i]), (y[i] - yc[i]), (z[i] - zc[i])])
            vel[i] = LA.norm([xv[i], yv[i], zv[i]])
            swing[i] = LA.norm([(xc[i] - pl[i][0]), (yc[i] - pl[i][1]), (zc[i] - 100 - pl[i][2])])
            # dist[i] = LA.norm([(x[i] - xc[i]), (y[i] - yc[i])])
            # dist[i] = LA.norm([(x[i] - xc[i]), (z[i] - zc[i])])

        plt.figure()
        plt.plot(s, swing)
        plt.axhline(y=150, color='b', linestyle=':')

        plt.figure()
        plt.plot(s[:390], x[:390], label = "Desired Waypoint")
        plt.plot(s[:390], xc[:390], label = "Leader Drone")
        plt.plot(s[:390], xc1[:390], label = "Follower Drone 1")
        plt.plot(s[:390], xc2[:390], label = "Follower Drone 2")
        plt.plot(s[:390], xc3[:390], label = "Follower Drone 3")
        plt.plot(s[:390], pl[:390,0], label="Payload")
        plt.xlabel("Steps")
        plt.ylabel("Distance")
        plt.title("X-axis")
        plt.legend(loc = 'center left')

        plt.figure()
        plt.plot(s[:390], y[:390])
        plt.plot(s[:390], yc[:390])
        plt.plot(s[:390], yc1[:390])
        plt.plot(s[:390], yc2[:390])
        plt.plot(s[:390], yc3[:390])
        plt.plot(s[:390], pl[:390, 1])
        plt.xlabel("Steps")
        plt.ylabel("Distance")
        plt.title("Y-axis")

        plt.figure()
        plt.plot(s[:390], z[:390])
        plt.plot(s[:390], zc[:390])
        plt.plot(s[:390], zc1[:390])
        plt.plot(s[:390], zc2[:390])
        plt.plot(s[:390], zc3[:390])
        plt.plot(s[:390], pl[:390, 2])
        plt.xlabel("Steps")
        plt.ylabel("Distance")
        plt.title("Z-axis")


        plt.figure()
        plt.plot(s[:390], dist[:390])
        plt.axhline(y=0, color='b', linestyle=':')
        plt.axhline(y=150, color='b', linestyle=':', label = "Target Boundary")


        plt.xlabel("Steps")
        plt.ylabel("Distance")
        plt.title("Total distance to the waypoint")
        plt.legend()

        # plt.figure()
        # plt.plot(rew_comps)
        # plt.xlabel("Steps")
        # plt.ylabel("Velocity")
        # plt.title("Total velocity")

        # plt.figure()
    # plt.hist(dataY[:,0], bins = 200)
    # plt.figure()
    # plt.hist(dataY[:, 1], bins=200)
    # # plt.figure()
    # # plt.hist(dataY[:, 2], bins=20)
    # plt.figure()
    # plt.hist(dataY[:, 2], bins=20)
    #
    # plt.figure()
    # plt.plot(dataY[:, 11])
    # plt.hist(dataX[:,0], bins = 20)
    # plt.hist(dataY1[:,0], bins = 20)





    # plt.plot(s, dataX)
    # plt.plot(x, z-10, linestyle='dashed')
    # plt.plot(x, z+10, linestyle='dashed')

    # plt.ylim([500,2000])


    plt.show()
    dataY_new = np.load(save_dir + '/training_data/dataY_new_iter2.npy')
    dataZ_new = np.load(save_dir + '/training_data/dataZ.npy')

    pass

main()
