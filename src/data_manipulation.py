import numpy as np
import numpy.random as npr
# import tensorflow as tf
import time
import math
# import matplotlib.pyplot as plt
import copy


def generate_training_data_inputs(states0, controls0):
    # init vars
    # states=np.copy(states0)
    states = list(states0)
    controls = list(controls0)
    # controls=np.copy(controls0)
    new_states = []
    new_controls=[]

    # remove the last entry in each rollout (because that entry doesn't have an associated "output")
    for i in range(len(states)):
        curr_item = np.copy(states[i])
        length = curr_item.shape[0]
        new_states.append(curr_item[20:length-21,:])

        curr_item = np.copy(controls[i])
        length = curr_item.shape[0]
        new_controls.append(curr_item[20:length-21,:])
   
    #turn the list of rollouts into just one large array of data
    dataX= np.concatenate(new_states, axis=0)
    dataY= np.concatenate(new_controls, axis=0)
    return dataX, dataY

def generate_training_data_outputs(states):
    #for each rollout, the output corresponding to each (s_i) is (s_i+1 - t
    differences = []
    for states_in_single_rollout in states:
        output = states_in_single_rollout[21:states_in_single_rollout.shape[0]-20,:]\
                -states_in_single_rollout[20:states_in_single_rollout.shape[0]-21,:]
        differences.append(output)
    output = np.concatenate(differences, axis=0)
    return output

def preproccessing_data(dataX, dataY, dataZ):
    mean_x = np.mean(dataX, axis=0)
    dataX = dataX - mean_x
    std_x = np.std(dataX, axis=0)
    dataX = np.nan_to_num(dataX / std_x)

    mean_y = np.mean(dataY, axis=0)
    dataY = dataY - mean_y
    std_y = np.std(dataY, axis=0)
    dataY = np.nan_to_num(dataY / std_y)

    mean_z = np.mean(dataZ, axis=0)
    dataZ = dataZ - mean_z
    std_z = np.std(dataZ, axis=0)
    dataZ = np.nan_to_num(dataZ / std_z)

    ## concatenate state and action, to be used for training dynamics
    NN_inputs = np.concatenate((dataX, dataY), axis=1)
    NN_outputs = np.copy(dataZ)
    return mean_x, mean_y, mean_z, std_x, std_y, std_z, NN_inputs, NN_outputs