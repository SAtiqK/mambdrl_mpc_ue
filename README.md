# Multi-Agent Model-Based Deep Reinforcement Learning for Collaborative Payload Transportation using UAVs

## Overview

This ROS package enables coordinated control of multiple UAVs for collaborative payload transport using PETS (Probabilistic Ensembles with Trajectory Sampling) with MPC (Model Predictive Control) and formation control. The system controls UAVs in an Unreal Engine simulated environment, supporting both single and multi-agent modes and an optional PID control.

---

## System Requirements

- **ROS**: Noetic  
- **Python**: 3.9  
- **Unreal Engine**: 4.27  
- **rosbridge_suite**: for communication with Unreal Engine  

---

## How to Run

### 1. Set up ROS-Unreal Engine Integration

 Follow instructions at [https://github.com/code-iai/ROSIntegration](https://github.com/code-iai/ROSIntegration).

### 2. Set Parameters

Edit the YAML config file `src/UEdrone_train.yaml`  to set running paramters. For example:

```yaml
use_pets: true         # Enable PETS-based dynamics modeling instead of vanilla MBDRL
multi_agent: true      # Use multi-agent or single agent control
use_pid: false         # Use PID instead of MBDRL (for initial data collection)
```

### 3. Start ROS Core
Run the following command on the machine intended to be the ROS Master:
```bash 
roscore
```

### 4. Launch Unreal Engine and rosbridge
Launch the [Unreal Engine simulation](https://github.com/SAtiqK/UAVs_PL_sim?tab=readme-ov-file).

Connect the simualtion with the ROS Master using rosbridge:
```bash
roslaunch rosbridge_server rosbridge_tcp.launch bson_only_mode:=True
```

### 5. Run UAV and Payload Nodes
Run nodes for each of the UAVs (four in this case) and the payload in the simulation by running the following scripts:
- `Node_drone1.py`
- `Node_drone2.py`
- `Node_drone3.py`
- `Node_drone4.py`
- `Node_pl.py` 

### 6. Run the main script (the controller)
Now that UAVs' and payload's states are available, run the controller to output and publish control inputs by running the following script:
- `mpc_execute.py`



## License
This project is licensed under the MIT License - see the LICENSE file for details.

