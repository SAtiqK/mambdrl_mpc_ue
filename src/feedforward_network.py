
import numpy as np
import tensorflow as tf

import tensorflow as tf

def feedforward_network(inputState, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype):

    initializer = tf.keras.initializers.GlorotUniform()  # Xavier initializer
    h_i = inputState  # Start with the input

    # Create hidden layers with ReLU activation
    for i in range(num_fc_layers):
        h_i = tf.keras.layers.Dense(
            units=depth_fc_layers,
            activation='relu',
            kernel_initializer=initializer,
            bias_initializer='zeros',
            dtype=tf_datatype
        )(h_i)

    # Output layer without activation (e.g., for regression tasks)
    z = tf.keras.layers.Dense(
        units=outputSize,
        activation=None,  # No activation for the output layer
        kernel_initializer=initializer,
        bias_initializer='zeros',
        dtype=tf_datatype
    )(h_i)

    return z


# def feedforward_network(inputState, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype):
#
#     #vars
#     intermediate_size=depth_fc_layers
#     reuse= False
#     #changed
#     # initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf_datatype)
#     # initializer = tf.keras.initializers.GlorotUniform(seed=None)
#     # fc = tf.contrib.layers.fully_connected
#     # # make hidden layers
#     # for i in range(num_fc_layers):
#     #     if (i == 0):
#     #         fc_i = fc(inputState, num_outputs=intermediate_size, activation_fn=None,
#     #                   weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
#     #     else:
#     #         fc_i = fc(h_i, num_outputs=intermediate_size, activation_fn=None,
#     #                   weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
#     #     h_i = tf.nn.relu(fc_i)
#     #
#     # # make output layer
#     # z = fc(h_i, num_outputs=outputSize, activation_fn=None, weights_initializer=initializer,
#     #        biases_initializer=initializer, reuse=reuse, trainable=True)
#
#     # initializer = tf.keras.initializers.GlorotUniform(seed=None)
#     # inputState = tf.transpose(inputState)
#     # model = tf.keras.Sequential([
#     #     tf.keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializer, input_shape=(inputSize,)),
#     #     tf.keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializer, input_shape=(inputSize,)),
#     #     tf.keras.layers.Dense(units=outputSize, activation='relu', kernel_initializer=initializer,  dtype=tf.float64)
#     # ])
#     # # model.predict(inputState, steps=1)
#     #
#     # z = model.predict(inputState, steps = 1000)
#     # Compile the model with an optimizer and loss function
#     # model.compile(optimizer='adam',
#     #               loss='categorical_crossentropy',
#     #               metrics=['accuracy'])
#
#     intermediate_size = depth_fc_layers
#     initializer = tf.initializers.GlorotUniform()  # Xavier initializer in TensorFlow 2.x
#     h_i = inputState  # Initialize h_i with inputState
#
#     # Create hidden layers
#     for i in range(num_fc_layers):
#         # Define a dense (fully connected) layer
#         fc_i = tf.keras.layers.Dense(units=intermediate_size, activation=None, kernel_initializer=initializer,
#                                      bias_initializer=initializer, trainable=True)
#
#         # Apply the dense layer to h_i
#         h_i = fc_i(h_i)
#
#         # Apply ReLU activation
#         h_i = tf.nn.relu(h_i)
#
#     # Create the output layer
#     z = tf.keras.layers.Dense(units=outputSize, activation= None,kernel_initializer=initializer,bias_initializer=initializer,
#                               trainable=True)(h_i)
#
#     return z