import os

class data_save():
    def __init__(self, run_num):
        self.epi_observations = []
        self.epi_actions = []
        # self.epi_observations = []
        self.observations = []
        self.actions = []
        self.output= []
        self.rewards = []
        self.waypoint = []
        self.save_dir = 'run_' + str(run_num)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(self.save_dir + '/losses')
            os.makedirs(self.save_dir + '/models')
            os.makedirs(self.save_dir + '/saved_forwardsim')
            os.makedirs(self.save_dir + '/saved_trajfollow')
            os.makedirs(self.save_dir + '/training_data')


class data_save_mpc():
    def __init__(self, run_num):
        self.list_episode_observations = []
        self.list_episode_actions = []
        self.episode_reward = 0
        self.list_episode_rewards = []
        self.total_episode_reward = []
        self.episode_steps = []
        self.total_episode_steps = []
        self.starting_states = []
        self.selected_multiple_u = []
        self.resulting_multiple_x = []
        self.prev_rew = 0
        self.waypoint = []
        self.reward_comps = []
        self.save_dir = 'run_' + str(run_num)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_dir+ '/losses'):
            os.makedirs(self.save_dir + '/losses')
        if not os.path.exists(self.save_dir + '/models'):
            os.makedirs(self.save_dir + '/models')
        if not os.path.exists(self.save_dir + '/saved_forwardsim'):
            os.makedirs(self.save_dir + '/saved_forwardsim')
        if not os.path.exists(self.save_dir + '/saved_trajfollow'):
            os.makedirs(self.save_dir + '/saved_trajfollow')
        if not os.path.exists(self.save_dir + '/training_data'):
            os.makedirs(self.save_dir + '/training_data')
