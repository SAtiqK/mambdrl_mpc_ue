#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Quaternion
import rospy
from geometry_msgs.msg import Vector3

class visualize_rollout():
    def __init__(self, controls, counter_agg, save_dir, num_rollouts):
        self.state_list = controls
        self.counter_agg = counter_agg
        self.steps_per_epi = np.load(save_dir + '/training_data/episode_steps_iter_2.npy')
        self.wp = np.load('run_2003/training_data/mpc_waypoints_2.npy')
        self.num_rollouts_to_vis = num_rollouts
        self.rollout_count = 0
        self.step_count = 0
        self.wp_count = 0
        # assert self.steps_per_epi.shape[0] == len(controls)

    def visualize(self, node):

        # if self.step_count == 0 and :
        dwp = Vector3(self.wp[self.rollout_count][0], self.wp[self.rollout_count][1], self.wp[self.rollout_count][2])
        node.desired_waypoint.publish(dwp)
            # self.wp_count += 1
        if self.rollout_count < self.num_rollouts_to_vis:
            if self.step_count < self.steps_per_epi[self.rollout_count]:
                p = self.state_list[self.rollout_count][self.step_count][0:3]
                o = self.state_list[self.rollout_count][self.step_count][3:6]
                # action = self.controls_list[self.rollout_count][self.step_count]
                self.step_count += 1
                # print(self.step_count)
                # input = Quaternion()
                # input.x = action[0]
                # input.y = action[1]
                # input.w = action[2]
                # input.z = 0
            else:
                p = self.state_list[self.rollout_count][self.step_count - 1][0:3]
                o = self.state_list[self.rollout_count][self.step_count - 1][3:6]

                # input = Quaternion()
                # input.x = 0
                # input.y = 0
                # input.z = 0
                # input.w = 0
                print('Rollout ' + str(self.rollout_count) + ' done')
                self.step_count = 0
                self.rollout_count += 1
                # node.steps.publish(True)
            return p, o
        else:
            print('Done visualizing')
            rospy.signal_shutdown("Done")
        #publish to the steps topic to indicate the episode is over

            return 100, 100