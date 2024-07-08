#! /usr/bin/env python3
# -*- coding: utf-8 -*-

cuda = "cuda:0"
env_name = 'ned2'
policy = "Gaussian"

gamma = 0.99
tau = 0.005
lr = 0.001
alpha = 0.2

seed = 123456

hidden_size = 64
Success_Standard = 0.9

num_steps = 10000001
batch_size = 512
start_steps = 10000
max_episode_steps = 256
time_sleep_interval = 0.05

isExit_IfSuccessLearning = True #목표 달성 시(success rate 0.9이상일 때) 학습을 종료할 것인지

replay_size = num_steps #1000000
cuda = "cuda"
