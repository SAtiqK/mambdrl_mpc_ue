#!/usr/bin/env python3
import numpy as np
import json
import argparse
import yaml
import pathlib
import pandas as pd
from mpmath import zeros
from torch.nn.init import ones_


def main():
    parser = argparse.ArgumentParser()
    # the yaml file that has all the params required

    # parser.add_argument('--save_dir', type=str, default = "/mnt/nas_server/skhursheed/ue_train/run_2005")
    parser.add_argument('--save_dir', type=str, default = "/home/CASLab/catkin_ws/src/unreal_pckage-ros-main/Multi_Agent/run_2")

    # parser.add_argument('--save_dir', type=str, default = "/mnt/nas_server/skhursheed/ue_train/run_2004")
    args = parser.parse_args()
    sav_dir = str(pathlib.Path().resolve()) +'/run_2'
    dataX = np.load(sav_dir + '/training_data/dataX.npy')  # input1: state
    dataY = np.load(sav_dir + '/training_data/dataY.npy')  # input2: control
    dataZ = np.load(sav_dir + '/training_data/dataZ.npy')  # output: nextstate-state


    with open('UEdrone_train.yaml', 'r') as f:
        params = yaml.safe_load(f)

    yaw_actuation = params['flags']['yaw_actuation']
    MMpreproccess = params['flags']['MinMaxpreproc']

    if yaw_actuation == False:
        dataY = np.delete(dataY, 2, axis=1)

    #To just get the rest stopped data out of the way
    dataX = dataX[300:dataZ.shape[0]]
    dataY = dataY[300:dataZ.shape[0]]
    dataZ = dataZ[300:dataZ.shape[0]]
    # dataX = dataX[0:dataY.shape[0]]

    #if any actions is greater than 1. Shouldn't be since it is clipped
    condition_actions = np.any(dataY > 1, axis=1)
    dataY = dataY[~condition_actions]
    dataX = dataX[~condition_actions]
    dataZ = dataZ[~condition_actions]

    #if there is a big jump in the data indicating a delay
    condition_output = np.any(dataZ > 50, axis=1)
    dataY = dataY[~condition_output]
    dataX = dataX[~condition_output]
    dataZ = dataZ[~condition_output]

    #removing duplicate data
    dataX_pd = pd.DataFrame(np.copy(dataX))
    dataY_pd = pd.DataFrame(np.copy(dataY))
    dataZ_pd = pd.DataFrame(np.copy(dataZ))

    duplicated_X =dataX_pd.duplicated()
    dataX = dataX_pd[~duplicated_X].values
    dataY = dataY_pd[~duplicated_X].values
    dataZ = dataZ_pd[~duplicated_X].values


    # dataX = dataX_pd.drop_duplicates()
    # dataY = dataY_pd.drop_duplicates()
    # dataZ = dataZ_pd.drop_duplicates()


    #maybe add clustering

    # dataX_val = np.load(sav_dir + '/training_data/dataX_val.npy')  # input1: state
    # dataY_val = np.load(sav_dir + '/training_data/dataY_val.npy')  # input2: control
    # dataZ_val = np.load(sav_dir + '/training_data/dataZ_val.npy')  # output: nextstate-state

    mean_x = np.mean(dataX, axis=0)
    std_x = np.std(dataX, axis=0)

    mean_y = np.mean(dataY, axis=0)
    std_y = np.std(dataY, axis=0)
    # std_y = np.ones(4)

    mean_z = np.mean(dataZ, axis=0)
    std_z = np.std(dataZ, axis=0)

    #to remove out of bound data

    if MMpreproccess == True:
        min_position = -7000
        max_position = 3000
        min_orientation = -90
        max_orientation = 90
        min_difference_p = np.min(dataZ[:,0:3])
        min_index = np.argmax(dataZ)
        max_difference_p = np.max(dataZ[:,0:3])
        min_difference_o = np.min(dataZ[:, 3:6])
        max_difference_o = np.max(dataZ[:, 3:6])
        min_vel = np.min(dataX[:,6:9])
        max_vel = np.max(dataX[:,6:9])
        min_avel = np.min(dataX[:,9:12])
        max_avel = np.max(dataX[:,9:12])
        min_vel_diff = np.min(dataZ[:,6:9])
        max_vel_diff = np.max(dataZ[:,6:9])
        min_avel_diff = np.min(dataZ[:,9:12])
        max_avel_diff = np.max(dataZ[:,9:12])


        dataX_pos_pp = -1 + 2 * (dataX[:,0:3] - min_position) / (max_position - min_orientation)
        dataX_or_pp = -1 + 2 * (dataX[:,3:6] - min_orientation) / (max_orientation - min_orientation)
        dataX_vel_pp = -1 + 2 * (dataX[:,6:9] - min_vel) / (max_vel - min_vel)
        dataX_avel_pp =  -1 + 2 * (dataX[:,9:12] - min_avel) / (max_avel - min_avel)

        dataZ_pos_pp = -1 + 2 * (dataZ[:, 0:3] + 20) / (20 + 20)
        dataZ_or_pp = -1 + 2 * (dataZ[:, 3:6] - min_difference_o) / (max_difference_o - min_difference_o)
        dataZ_vel_pp = -1 + 2 * (dataZ[:, 6:9] - min_vel_diff) / (max_vel_diff - min_vel_diff)
        dataZ_avel_pp = -1 + 2 * (dataZ[:, 9:12] - min_avel_diff) / (max_avel_diff - min_avel_diff)

        dataX_pp = np.concatenate((dataX_pos_pp, dataX_or_pp, dataX_vel_pp, dataX_avel_pp), axis=1)
        dataZ_pp = np.concatenate((dataZ_pos_pp, dataZ_or_pp, dataZ_vel_pp, dataZ_avel_pp), axis=1)


    else:

        dataX = dataX - mean_x
        dataX_pp = np.nan_to_num(dataX / std_x)

        dataY = dataY - mean_y
        dataY_pp = np.nan_to_num(dataY / std_y)

        dataZ = dataZ - mean_z
        dataZ_pp = np.nan_to_num(dataZ / std_z)


    # dataY_pp = dataY

    # ny = dataY[:,0]*1000
    #
    # # Create a histogram
    # plt.hist(ny, bins=10000, edgecolor='k')  # 'bins' controls the number of bins in the histogram
    # plt.title('Data Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    #
    # # Display the plot
    # plt.show()

    np.save(sav_dir + '/training_data/dataX_pp.npy', dataX_pp)
    np.save(sav_dir + '/training_data/dataY_pp.npy', dataY_pp)
    np.save(sav_dir + '/training_data/dataZ_pp.npy', dataZ_pp)

    inputs = np.concatenate((dataX, dataY), axis=1)
    outputs = np.copy(dataZ)

    assert inputs.shape[0] == outputs.shape[0]
    inputSize = inputs.shape[1]
    outputSize = outputs.shape[1]

    data = {
        "inputSize": inputSize,
        "outputSize": outputSize,
        "mean_x": mean_x.tolist(),
        "mean_y": mean_y.tolist(),
        "mean_z": mean_z.tolist(),
        "std_x": std_x.tolist(),
        "std_y": std_y.tolist(),
        "std_z": std_z.tolist()
    }

    with open(sav_dir + "/training_data/data_stats.json", "w") as json_file:
        json.dump(data, json_file)


if __name__ == '__main__':
        main()
