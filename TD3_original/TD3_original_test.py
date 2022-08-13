
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
import torch
import os
from pathlib import Path

from gym_unity.envs import UnityEnv
from td3 import TD3Agent

# get_ipython().magic('matplotlib inline')

torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy


# In[ ]:
"-----------------------------------------1、设置环境--------------------------------------------"
## 训练时记得注释掉保存路径数据到csv，测试时保存数据
# env_name = "E:/RL/Unity-TD3/Env_Train/Test"  # Name of the Unity environment binary to launch
# env_name = "C:/Users/user110/Desktop/Unity-TD3-Mytest/Unity-TD3/Env_Test/Test"  # Name of the Unity environment binary to launch
env_name = os.path.abspath("./Env_Test/Test")
env_name = Path(env_name)
# env_name = "Macintosh HD:/system/Users/apple/Desktop/UnityProjects/Unity-TD3-Mytest/Unity-TD3/Env_Test/Test"
env = UnityEnv(str(env_name), worker_id=0)
env.seed(7)

# Examine environment parameters
print(str(env))
print(env.observation_space.shape[0])
print(env.action_space.shape[0])
print(env.action_space.low)
print(env.action_space.high)


# In[ ]:

## 环境测试
"-----------------------------2、环境测试------------------------------"
for episode in range(5):
    initial_observation = env.reset()
    episode_rewards = 0
    done = False
    while not done:
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        print(reward)
#         print(done)
        print(observation)

        episode_rewards += reward
        if done:
            text = 'Success' if reward>0 else 'Failed'
            print(text, episode_rewards)
            break
    print("Total reward this episode: {}".format(episode_rewards))


# In[ ]:
"------------------------------3、设置强化学习模型------------------------------"
gamma = 0.99
tau = 1e-2
delay_step = 2
policy_noise = 0.2
noise_bound = 0.5
max_action = 1
critic_lr = 1e-3
actor_lr = 1e-3
buffer_maxlen = 1000      # 100000
batch_size = 128

agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, policy_noise, noise_bound, max_action, critic_lr, actor_lr)


# In[ ]:

## 训练
"------------------------------4、训练------------------------------"
t1 = time.time()
max_episodes = 2000
expl_noise = 1
num = 0
scores = []
success = []

for i_episode in range(1, max_episodes+1):
    state = env.reset()
    score = 0
    steps = 0

    while True:
        action = np.clip((agent.get_action(state)
                          + np.random.normal(0, expl_noise, size=env.action_space.shape[0])),
                         env.action_space.low, env.action_space.high)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward / 10, next_state, done)
        num += 1
        score += reward / 10

        if len(agent.replay_buffer) == buffer_maxlen:
            agent.update(batch_size)
            expl_noise *= 0.999995

        if done:
            success += [reward>0]
            break

        state = next_state

    scores += [score]
    print('\rEpisode', i_episode, 'Average Score: %.2f'%np.mean(scores[-1:]),
          'Success rate %.2f'%np.mean(success[-20:]),
          'exploration noise std %.6f'%expl_noise,
          end="")

    agent.save(directory = './save_My./exp2./')

print('Running time: ', time.time() - t1)


# In[ ]:

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(scores[-6000:])
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[ ]:

plt.plot(success[-5000:])


# In[ ]:

## 保存scores到csv文件
name = ['Value']
reward_curve = pd.DataFrame(columns=name,data=scores)
reward_curve.to_csv('C:/Users/user110/Desktop/Unity-TD3-Mytest/Unity-TD3/rl/TD3_original/save_My/exp2/scores_td3_original.csv')

## 保存success到success_td3_original.csv文件
name = ['Value']
reward_curve = pd.DataFrame(columns=name,data=success)
reward_curve.to_csv('C:/Users/user110/Desktop/Unity-TD3-Mytest/Unity-TD3/rl/TD3_original/save_My/exp2/success_td3_original.csv')


# In[ ]:

## 测试2000个episode
## Time Scale=100，撞到障碍物结束
"------------------------------5、测试------------------------------"
agent.load(directory = './exp_Ob./')
scores = []
success = []
collision = []
lost = []

for i_episode in range(2000):
    state = env.reset()
    score = 0
    
    while True:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        score += reward / 10
        
        if done:
            success += [reward>0]
            collision += [reward<-50]
            lost += [reward>-50 and reward<0]
            scores += [score]
            text = 'Success' if reward>0 else 'Failed'
            print('\rEpisode',i_episode, text, ', Total Reward= %.2f'%score, 
                  ', Success rate %.5f'%np.mean(success[:]),
                  ', Collision rate %.5f'%np.mean(collision[:]),
                  ', Lost rate %.5f'%np.mean(lost[:]),
                  ', AR %.5f'%np.mean(scores[:]))
            break
            


# In[ ]:

env.close()


# In[ ]:



