
class episodeInfo():

    def __init__(self):
        pass

    prev_terminal = bool(0)
    episode_counter = 0
    episode_number = 10
    episode_val = -1
    episode_val_counter = 0
    steps_rollout = 500
    steps_rollout_counter = 0
    steps = bool(0)
    simu_status = bool(1)

class episodeInfoMPC():

    def __init__(self):
        pass

    prev_terminal = bool(0)
    episode_counter = 0
    episode_number = 2
    steps_rollout = 1000
    steps_rollout_counter = 0
    steps = bool(0)
    simu_status = bool(1)
