#!/usr/bin/env python3


from data_manipulation import preproccessing_data
import numpy as np
from dynamics_model import Dyn_Model
import tensorflow as tf
import os
import pickle
import argparse
import yaml
import json
# from validation_dynmodel import validation
import keras
import pandas as pd
import matplotlib.pyplot as plt
from PETS_model import nn_constructor
from dotmap import DotMap
import torch


# def dyn_model_get(existing_model, sess, tf_datatype, fraction_use_new, print_minimal):
def main():
    # if (counter_agg_iters == 0):
    print(tf.__version__)
    parser = argparse.ArgumentParser()
    # the yaml file that has all the params required
    parser.add_argument('--run_num', type=int, default=5006)
    parser.add_argument('--yaml_path', type=str,
                        # default='/mntnas_server/skhursheed/catkin_workspace/src/unreal_training/ue_train/UEdrone_train.yaml')
                        default='/home/CASLab/catkin_ws/src/unreal_pckage-ros-main/Multi_Agent/UEdrone_train.yaml')
    parser.add_argument('--counter_agg', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='run_5006')

    args = parser.parse_args()
    counter_agg = args.counter_agg
    run_number = args.run_num

    with open('UEdrone_train.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Dynamic Model parameters
    lr = params['dyn_model']['lr']
    batchsize = params['dyn_model']['batchsize']
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    tf_datatype = tf.float64
    PETS = params['flags']['use_PETS']

    with open(args.save_dir + "/training_data/data_stats.json", "r") as json_file:
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


    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
#<<<<<<< HEAD
#=======
    gpu_device = 1
    gpu_frac = 0.9
#>>>>>>> a0465618a9a1bcee1d4b6d6aee9813ef04e6ba44

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                      log_device_placement=False,
                                      allow_soft_placement=True,
                                      inter_op_parallelism_threads=1,
                                      intra_op_parallelism_threads=1)

    # with tf.compat.v1.Session(config=config) as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())

    if PETS:
        model_init_config = DotMap(num_nets=3, model_in=inputSize, model_out=outputSize)
        dyn_model = nn_constructor(model_init_config)
    else:
        dyn_model = Dyn_Model(inputSize, outputSize, 1, lr, batchsize, num_fc_layers,
                          depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal = False)

    # saver = tf.compat.v1.train.Saver(max_to_keep=5) #WAS 0 BEFORE
    existing_model = params['flags']['use_existing_dyn_model']
    # all from the parser
    num_fc_layers= 1
    depth_fc_layers= 500
    batchsize= 512
    lr= 0.001
    nEpoch= params['dyn_model']['nEpoch']
#<<<<<<< HEAD
    fraction_use_new= params['dyn_model']['fraction_use_new']
#=======
    fraction_use_new= 0
#>>>>>>> a0465618a9a1bcee1d4b6d6aee9813ef04e6ba44
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    gpu_device = 1
    gpu_frac = 0.9
    tf_datatype = tf.float64
    print_minimal = False

    save_dir = 'run_' + str(run_number)
    dataX = np.load(save_dir + '/training_data/dataX_pp.npy')  # input1: state
    dataY = np.load(save_dir + '/training_data/dataY_pp.npy')  # input2: control
    dataZ = np.load(save_dir + '/training_data/dataZ_pp.npy')  # output: nextstate-state
    # dataX = dataX[0:1000000][:]
    # dataY = dataY[0:1000000][:]
    # dataZ = dataZ[0:1000000][:]

    inputs = np.concatenate((dataX, dataY), axis=1)
    outputs = np.copy(dataZ)

    assert inputs.shape[0] == outputs.shape[0]
    inputSize = inputs.shape[1]
    outputSize = outputs.shape[1]


    if counter_agg > 0:
        dataX_new = np.load(save_dir + '/training_data/dataX_new_iter' + str(counter_agg) + '.npy') # (0, observation space size)
        dataY_new = np.load(save_dir + '/training_data/dataY_new_iter' + str(counter_agg) + '.npy')
        dataZ_new = np.load(save_dir + '/training_data/dataZ_new_iter' + str(counter_agg) + '.npy')
    else:
        dataX_new = np.zeros((0, mean_x.shape[0]))  # (0, observation space size)
        dataY_new = np.zeros((0, mean_y.shape[0]))
        dataZ_new = np.zeros((0, mean_z.shape[0]))
    # training_loss_list = np.load(save_dir + '/losses/list_training_loss.npy')
    # old_loss_list = np.load(save_dir + '/losses/list_old_loss.npy')
    # new_loss_list = np.load(save_dir + '/losses/list_new_loss.npy')

    training_loss_list = []
    old_loss_list = []
    new_loss_list = []
    ## concatenate state and action, to be used for training dynamics


    # mean_x, mean_y, mean_z, std_x, std_y, std_z, inputs, outputs, = preproccessing_data(dataX, dataY, dataZ)

    assert inputs.shape[0] == outputs.shape[0]
    inputSize = inputs.shape[1]
    outputSize = outputs.shape[1]

    if (not (print_minimal)):
        print("\n#####################################")
        print("Preprocessing 'new' training data")
        print("#####################################\n")

    dataX_new = dataX_new[10:dataX_new.shape[0]]
    dataY_new = dataY_new[10:dataY_new.shape[0]]
    dataZ_new = dataZ_new[10:dataZ_new.shape[0]]

    condition_actions = np.any(dataY_new > 1, axis=1)
    dataY_new = dataY_new[~condition_actions]
    dataX_new = dataX_new[~condition_actions]
    dataZ_new = dataZ_new[~condition_actions]

    condition_output = np.any(abs(dataZ_new) > 50, axis=1)

    # Finding the indices where the condition is True
    true_indices = np.where(condition_output)[0]

    # Expanding the range of indices to remove five rows above and below the true indices
    rows_to_remove = []
    for idx in true_indices:
        rows_to_remove.extend(list(range(max(0, idx - 1), min(len(dataZ_new), idx + 1))))

    # Removing duplicates and sorting the rows to remove
    rows_to_remove = sorted(list(set(rows_to_remove)))

    # Filtering data based on rows to remove
    dataY_new = np.delete(dataY_new, rows_to_remove, axis=0)
    dataX_new = np.delete(dataX_new, rows_to_remove, axis=0)
    dataZ_new = np.delete(dataZ_new, rows_to_remove, axis=0)

    # dataY_new = dataY_new[~condition_output]
    # dataX_new = dataX_new[~condition_output]
    # dataZ_new = dataZ_new[~condition_output]

    dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x) / std_x)
    # dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y) / std_y)
    dataY_new_preprocessed = dataY_new
    dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z) / std_z)

    dataX_pd = pd.DataFrame(np.copy(dataX_new_preprocessed))
    dataY_pd = pd.DataFrame(np.copy(dataY_new_preprocessed))
    dataZ_pd = pd.DataFrame(np.copy(dataZ_new_preprocessed))

    duplicated_X = dataX_pd.duplicated()
    dataX_new_preprocessed = dataX_pd[~duplicated_X].values
    dataY_new_preprocessed = dataY_pd[~duplicated_X].values
    dataZ_new_preprocessed = dataZ_pd[~duplicated_X].values



    inputs_new = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1)
    outputs_new = np.copy(dataZ_new_preprocessed)
    # with open("models/dyn_model_" + str(run_number) + "_instance" + ".pkl", "rb") as file:
    #     dyn_model = pickle.load(file)
    if counter_agg > 0:
        existing_model = False

    if existing_model == True:
        # restore_path = save_dir + '/models/finalModel.ckpt'
        # saver.restore(sess, restore_path)
        # print("Model restored from ", restore_path)
        model = keras.models.load_model(save_dir + '/models/finalModel' + str(counter_agg))
        dyn_model.curr_nn_output = model
        training_loss = 0
        old_loss = 0
        new_loss = 0
    elif PETS:
        print("Training PETS")
        training_loss, old_loss, new_loss = dyn_model.train(inputs, outputs, inputs_new, outputs_new, nEpoch)
        torch.save(dyn_model.state_dict(), save_dir + '/models/finalModel.pth')

    else:
        print('here')
        training_loss, old_loss, new_loss = dyn_model.train(inputs, outputs, inputs_new, outputs_new,
                                                            nEpoch, save_dir, fraction_use_new)

    # how good is model on training data
    # training_loss_list.tolist().append(training_loss)
    # # how good is model on old dataset
    # old_loss_list.tolist().append(old_loss)
    # # how good is model on new dataset
    # new_loss_list.tolist().append(new_loss)

    training_loss_list.append(training_loss)
    # how good is model on old dataset
    old_loss_list.append(old_loss)
    # how good is model on new dataset
    new_loss_list.append(new_loss)
    print("Training Loss: " + str(training_loss))

    np.save(save_dir + '/losses/list_training_loss.npy', training_loss_list)
    np.save(save_dir + '/losses/list_old_loss.npy', old_loss_list)
    np.save(save_dir + '/losses/list_new_loss.npy', new_loss_list)

    # with open(save_dir + "/models/dyn_model_" + str(run_number) + "_instance" + ".pkl", "wb") as file:
    #     pickle.dump(dyn_model, file)

    # save_path = saver.save(sess, save_dir + '/models/model_aggIter' + str(counter_agg) + '.ckpt')
    # save_path = saver.save(sess, save_dir + '/models/finalModel.ckpt')
    # if (not (print_minimal)):
    #     print("Model saved at ", save_path)
    w = False
    if w:
        error_1step, error_5step, error_10step, error_50step, error_100step = validation(dyn_model, dyn_model.curr_nn_output, mean_x, std_x)

        if counter_agg == 0:
            errors_1_per_agg = []
            errors_5_per_agg = []
            errors_10_per_agg = []
            errors_50_per_agg = []
            errors_100_per_agg = []
        else:
            errors_1_per_agg = np.load(save_dir + '/errors_1_per_agg.npy')
            errors_5_per_agg = np.load(save_dir + '/errors_5_per_agg.npy')
            errors_10_per_agg = np.load(save_dir + '/errors_10_per_agg.npy')
            errors_50_per_agg = np.load(save_dir + '/errors_50_per_agg.npy')
            errors_100_per_agg = np.load(save_dir + '/errors_100_per_agg.npy')

        errors_1_per_agg.append(error_1step)
        errors_5_per_agg.append(error_5step)
        errors_10_per_agg.append(error_10step)
        errors_50_per_agg.append(error_50step)
        errors_100_per_agg.append(error_100step)

        np.save(save_dir + '/errors_1_per_agg.npy', errors_1_per_agg)
        np.save(save_dir + '/errors_5_per_agg.npy', errors_5_per_agg)
        np.save(save_dir + '/errors_10_per_agg.npy', errors_10_per_agg)
        np.save(save_dir + '/errors_50_per_agg.npy', errors_50_per_agg)
        np.save(save_dir + '/errors_100_per_agg.npy', errors_100_per_agg)


if __name__ == '__main__':
    main()

