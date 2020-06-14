# Core RL

![Python package](https://github.com/djbyrne/core_rl/workflows/Python%20package/badge.svg?branch=master)

This is my personal repo containing the algorithms I am working on to further my understanding of RL. This repo contains
many of the core RL algorithms implemented with the 
[Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework. 

Most of these algorithms were based off the implementations found in 
[Deep Reinforcement Learning Hands On: Second Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)

## Algorithms

### Off Policy
- [X] DQN
- [X] Double DQN
- [X] Dueling DQN
- [X] Noisy DQN
- [X] DQN with Prioritized Experience Replay
- [X] N Step DQN
- [ ] DDPG
- [ ] TD3
- [ ] SAC

### On Policy
- [ ] REINFORCE/Vanilla Policy Gradient
- [ ] A3C
- [ ] A2C
- [ ] PPO
- [ ] GAIL

## Training Features
- [ ] LSTM heads
- [ ] Augmented Data
- [ ] Mixed Precision
- [ ] Discriminate Learning Rates


## Installation

````bash

conda create -n core_rl python==3.7

conda activate core_rl

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt

````

## Quick Start
Each algorithm implemented has a folder in the algos directory. This contains the Lightning module for each algorith. To train a model, exec the run.py file and specify the algorithm and environment you want to run.

```bash
python run.py --algo dqn --env PongNoFrameskip-v4
```

---
**NOTE**

Currently, the DQN models are hard coded to use a CNN and only work out of the box for the Atari environment. Similarly, the policy gradient methods currently only have a MLP net and is really only tested on the classic control tasks. This will be updated soon!

---
