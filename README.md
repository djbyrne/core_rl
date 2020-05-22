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
- [ ] DDPG
- [ ] TD3
- [ ] SAC

### On Policy
- [ ] REINFORCE
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